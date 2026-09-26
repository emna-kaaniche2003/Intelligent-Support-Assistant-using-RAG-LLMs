"""
ai_engine/rag/evaluate_retrieval.py
Évaluation du retrieval hybride (dense + BM25 sparse) sur le golden dataset,
contre la collection Qdrant "connaissances_modules".

Usage :
    python ai_engine/rag/evaluate_retrieval.py
    python ai_engine/rag/evaluate_retrieval.py --golden tests/golden_dataset.json \
        --ingestion-config data_pipeline/config/ingestion_config.json

Fusion hybride — écart documenté avant écriture du code :
qdrant-client==1.9.2 (celui installé ici, cf. requirements.txt, doit matcher le
serveur Docker) n'a PAS l'API Query unifiée avec prefetch + fusion RRF côté serveur
(vérifié par introspection : pas de `QdrantClient.query_points`, pas de
`models.Prefetch`/`models.FusionQuery`). Cette API n'existe qu'à partir de
qdrant-client ~1.10+/1.12+. Ce qui EST disponible en 1.9.2 : `search()` par vecteur
nommé et `search_batch()` (plusieurs requêtes de recherche en un seul appel réseau).
On effectue donc une recherche dense ET une recherche sparse via `search_batch()`,
puis on fusionne les deux classements côté client par Reciprocal Rank Fusion (RRF :
score = somme de 1/(k+rang) sur les listes où le point apparaît, k=60, constante
standard de la littérature RRF).

Profondeur d'évaluation — un seul appel hybrid_search() par question :
on remonte TOP_K_MAX (10) résultats fusionnés en un seul appel, puis
recall@1/@3/@5/@10 sont calculés en tronquant CETTE MÊME liste à différentes
profondeurs (jamais 4 appels avec des `limit` différents : la fusion RRF n'a
aucune garantie de stabilité inter-appels si elle était relancée à limit variable).

Garde-fou vérité-terrain : avant de scorer une question, on vérifie via
client.retrieve() que le point ground_truth_id référencé par id_reference
existe réellement dans la collection. S'il est absent (référence invalide dans
le golden dataset — donnée non trouvée, pas un échec de retrieval), la question
est exclue du calcul de recall/precision/MRR et comptée séparément.

Réutilisation (pas de duplication) depuis data_pipeline/ingestion/ingest_qr_technique.py :
- build_point_id()   : calcul de l'UUID de vérité terrain, identique à l'ingestion
- tokenize()/build_bm25_index()/vectorize_bm25() : même vocabulaire/IDF BM25 que
  l'ingestion (le corpus de référence est reconstruit à l'identique via
  load_config/load_source/prepare_entries, pour que les indices sparse du vecteur
  requête correspondent aux mêmes dimensions que celles utilisées à l'ingestion)
- DENSE_VECTOR_NAME/SPARSE_VECTOR_NAME : noms de vecteurs de la collection
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import NamedSparseVector, NamedVector, SearchRequest
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_engine.config import get_settings
from data_pipeline.ingestion.ingest_qr_technique import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    build_bm25_index,
    build_point_id,
    load_config as load_ingestion_config,
    load_source,
    prepare_entries,
    tokenize,
    vectorize_bm25,
)

DEFAULT_GOLDEN_PATH = "tests/golden_dataset.json"
DEFAULT_INGESTION_CONFIG_PATH = "data_pipeline/config/ingestion_config.json"
DEFAULT_REPORT_PATH = "tests/retrieval_eval_report.json"

RRF_K = 60          # constante standard RRF (Cormack et al. 2009)
FETCH_LIMIT = 20     # nb de résultats remontés par sous-recherche (dense/sparse) avant fusion
RECALL_DEPTHS = [1, 3, 5, 10]  # profondeurs de recall évaluées sur une seule liste fusionnée
TOP_K_MAX = max(RECALL_DEPTHS)  # 10 : profondeur unique remontée par question
ECHEC_INSPECTION_TOP_N = 3      # nb de résultats affichés en détail pour chaque échec

# Poids RRF par défaut = 1.0/1.0 (équivalent à la fusion non pondérée d'origine :
# multiplier les deux classements par le même facteur ne change pas l'ordre relatif).
# Poids configurables (jamais figés en dur au-delà de ce défaut neutre) pour permettre
# la recherche en grille (grid_search_rrf_weights) sans modifier le comportement
# de production tant qu'aucun poids n'a été validé.
DEFAULT_W_DENSE = 1.0
DEFAULT_W_SPARSE = 1.0
DEFAULT_GRID_W_DENSE = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_GRID_REPORT_PATH = "tests/retrieval_grid_search_report.json"


# ── Chargement ────────────────────────────────────────────────────────────────

GOLDEN_DATASET_FILENAME_PATTERN = re.compile(r"^golden_dataset(?:_v(\d+))?\.json$")
DEFAULT_GOLDEN_DATASET_DIR = "tests"


def _golden_dataset_version(filename: str) -> int | None:
    """
    Extrait le numéro de version depuis un nom de fichier golden_dataset*.json.
    `golden_dataset.json` (sans suffixe) → version 0 (priorité minimale).
    `golden_dataset_v2.json` → version 2. Retourne None si le nom ne correspond
    pas au motif attendu (fichier ignoré par resolve_golden_dataset).
    """
    match = GOLDEN_DATASET_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    return int(match.group(1)) if match.group(1) else 0


def resolve_golden_dataset(
    override_path: str | None = None, search_dir: str = DEFAULT_GOLDEN_DATASET_DIR
) -> str:
    """
    Détermine le chemin du golden dataset à utiliser :
    - override_path fourni (--dataset-path) : utilisé tel quel, aucune recherche.
    - sinon : liste les fichiers `golden_dataset*.json` de `search_dir`, extrait
      leur suffixe de version (`golden_dataset.json` traité comme v0), et
      sélectionne le fichier de version la plus élevée.
    """
    if override_path:
        return override_path

    try:
        filenames = os.listdir(search_dir)
    except FileNotFoundError:
        filenames = []

    versioned = [
        (version, name)
        for name in filenames
        if (version := _golden_dataset_version(name)) is not None
    ]
    if not versioned:
        raise FileNotFoundError(
            f"Aucun fichier golden_dataset*.json trouvé dans {search_dir}"
        )

    versioned.sort(key=lambda vp: vp[0])
    _, chosen_name = versioned[-1]
    return f"{search_dir}/{chosen_name}"


def log_golden_dataset_selection(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nb_questions = len(data.get("questions", []))
    version = _golden_dataset_version(Path(path).name)
    version_label = f"v{version}" if version is not None else "inconnue"
    print(f"[INFO] Golden dataset chargé : {path} (version {version_label}, {nb_questions} questions)")


def load_golden_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def build_reference_corpus(ingestion_config_path: str) -> tuple[list[str], dict[str, Any]]:
    """
    Reconstruit exactement le corpus texte utilisé par ingest_qr_technique.py
    (mêmes fonctions : load_config/load_source/prepare_entries), pour que le
    vocabulaire BM25 (et donc les indices du vecteur sparse requête) soit
    strictement identique à celui utilisé lors de l'ingestion des 775 points.
    """
    config = load_ingestion_config(ingestion_config_path)
    raw_entries = load_source(config["chemin_source"])
    _, texts, _ = prepare_entries(raw_entries, config)
    return texts, config


# ── Fusion hybride (RRF côté client, cf. docstring module) ───────────────────

def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = RRF_K,
    w_dense: float = DEFAULT_W_DENSE,
    w_sparse: float = DEFAULT_W_SPARSE,
) -> list[tuple[str, float]]:
    """
    Fusionne deux classements Qdrant (ScoredPoint) par Reciprocal Rank Fusion
    pondérée : score = w_dense/(k+rang_dense) + w_sparse/(k+rang_sparse).
    Poids par défaut 1.0/1.0 = comportement non pondéré d'origine (seul le
    ratio w_dense/w_sparse influence l'ordre final, pas leur magnitude absolue).
    Retourne [(point_id, score_fusionné), ...] trié par score décroissant.
    """
    scores: dict[str, float] = {}
    for rank, hit in enumerate(dense_results, start=1):
        point_id = str(hit.id)
        scores[point_id] = scores.get(point_id, 0.0) + w_dense / (k + rank)
    for rank, hit in enumerate(sparse_results, start=1):
        point_id = str(hit.id)
        scores[point_id] = scores.get(point_id, 0.0) + w_sparse / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def fetch_dense_sparse_candidates(
    client: QdrantClient,
    collection_name: str,
    dense_query: list[float],
    sparse_query,
    limit: int = FETCH_LIMIT,
) -> tuple[list, list]:
    """
    Un seul appel search_batch() (dense + sparse), sans fusion. Extrait de
    hybrid_search() pour permettre au grid search de poids RRF de ne récupérer
    le pool de candidats qu'UNE SEULE FOIS par question, puis de refusionner en
    mémoire pour chaque combinaison de poids testée (pas de requête réseau
    supplémentaire par poids).
    """
    requests = [
        SearchRequest(
            vector=NamedVector(name=DENSE_VECTOR_NAME, vector=dense_query),
            limit=limit,
            with_payload=False,
        ),
        SearchRequest(
            vector=NamedSparseVector(name=SPARSE_VECTOR_NAME, vector=sparse_query),
            limit=limit,
            with_payload=False,
        ),
    ]
    dense_results, sparse_results = client.search_batch(
        collection_name=collection_name, requests=requests
    )
    return dense_results, sparse_results


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    dense_query: list[float],
    sparse_query,
    limit: int,
    w_dense: float = DEFAULT_W_DENSE,
    w_sparse: float = DEFAULT_W_SPARSE,
) -> list[tuple[str, float]]:
    """
    Recherche hybride réelle : un seul appel search_batch() (dense + sparse),
    fusion RRF pondérée côté client. Retourne le top `limit` fusionné.
    """
    dense_results, sparse_results = fetch_dense_sparse_candidates(
        client, collection_name, dense_query, sparse_query, limit=FETCH_LIMIT
    )
    fused = reciprocal_rank_fusion(dense_results, sparse_results, w_dense=w_dense, w_sparse=w_sparse)
    return fused[:limit]


# ── Évaluation par question ───────────────────────────────────────────────────

def evaluate_question(
    question: dict[str, Any],
    model: SentenceTransformer,
    bm25,
    vocab: dict[str, int],
    client: QdrantClient,
    collection_name: str,
    w_dense: float = DEFAULT_W_DENSE,
    w_sparse: float = DEFAULT_W_SPARSE,
) -> dict[str, Any]:
    """
    Évalue une question du golden dataset : recall@1/3/5/10, precision@1, MRR.

    Garde-fou vérité-terrain : si le point ground_truth_id (dérivé de
    id_reference) n'existe pas dans la collection, la question est exclue
    (reference_invalide=True) et n'est PAS comptée comme un échec de retrieval.
    """
    ground_truth_id = build_point_id(question["id_reference"], question["source_type"])

    existants = client.retrieve(collection_name=collection_name, ids=[ground_truth_id])
    if not existants:
        return {
            "id_test": question["id_test"],
            "categorie": question["categorie"],
            "interface_ou_serveur": question["interface_ou_serveur"],
            "reference_invalide": True,
        }

    query_text = question["question_client"]
    dense_query = model.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    sparse_query = vectorize_bm25(tokenize(query_text), bm25, vocab)

    # Un seul appel hybrid_search (top TOP_K_MAX) : recall@1/3/5/10 dérivés
    # par troncature de cette même liste fusionnée, jamais par ré-appels
    # séparés à limit variable (pas de garantie de stabilité RRF inter-appels).
    top_results = hybrid_search(
        client, collection_name, dense_query, sparse_query, limit=TOP_K_MAX,
        w_dense=w_dense, w_sparse=w_sparse,
    )
    top_ids = [point_id for point_id, _ in top_results]

    trouve = ground_truth_id in top_ids
    rang_trouve = top_ids.index(ground_truth_id) + 1 if trouve else None
    precision_1 = 1 if top_ids and top_ids[0] == ground_truth_id else 0
    score_premier = top_results[0][1] if top_results else None
    reciprocal_rank = 1.0 / rang_trouve if rang_trouve is not None else 0.0

    recalls = {
        f"recall_at_{profondeur}": 1 if (rang_trouve is not None and rang_trouve <= profondeur) else 0
        for profondeur in RECALL_DEPTHS
    }

    return {
        "id_test": question["id_test"],
        "categorie": question["categorie"],
        "interface_ou_serveur": question["interface_ou_serveur"],
        "reference_invalide": False,
        "trouve_top3": recalls["recall_at_3"] == 1,
        "rang_trouve": rang_trouve,
        **recalls,
        "precision_at_1": precision_1,
        "reciprocal_rank": reciprocal_rank,
        "score_du_premier_resultat": score_premier,
        "ground_truth_id": ground_truth_id,
        "top_ids": top_ids,
    }


# ── Inspection manuelle des échecs (payload à la demande, pas dans hybrid_search) ─

def enrichir_echecs_avec_payload(
    client: QdrantClient,
    collection_name: str,
    results: list[dict[str, Any]],
    top_n: int = ECHEC_INSPECTION_TOP_N,
) -> None:
    """
    Pour chaque question ratée (recall@3=0, référence valide), récupère le
    payload (categorie/question/reponse tronquée à 100 car.) de la vérité-terrain
    et des `top_n` résultats retournés, pour permettre l'inspection manuelle.

    Modifie `results` en place (ajoute "verite_terrain_resume"/"top_resultats_resume"
    sur les questions en échec uniquement). Un seul appel retrieve() batché,
    plutôt que with_payload=True sur la recherche dense elle-même : hybrid_search()
    ne renvoie que des tuples (point_id, score) — cf. signature imposée pour
    l'extraction vers retriever.py — donc le payload est récupéré séparément ici,
    à la demande, uniquement pour les cas qu'on a réellement besoin d'inspecter.
    """
    echecs = [r for r in results if not r.get("reference_invalide") and r["recall_at_3"] == 0]
    if not echecs:
        return

    ids_a_recuperer: set[str] = set()
    for r in echecs:
        ids_a_recuperer.add(r["ground_truth_id"])
        ids_a_recuperer.update(r["top_ids"][:top_n])

    points = client.retrieve(
        collection_name=collection_name, ids=list(ids_a_recuperer), with_payload=True
    )
    payload_par_id = {str(p.id): p.payload for p in points}

    def resume(point_id: str) -> dict[str, Any] | None:
        payload = payload_par_id.get(point_id)
        if not payload:
            return None
        return {
            "categorie": payload.get("categorie"),
            "question": (payload.get("question") or "")[:100],
            "reponse": (payload.get("reponse") or "")[:100],
        }

    for r in echecs:
        r["verite_terrain_resume"] = resume(r["ground_truth_id"])
        r["top_resultats_resume"] = [resume(pid) for pid in r["top_ids"][:top_n]]


# ── Pipeline principal ────────────────────────────────────────────────────────

def _moyenne(scores: list[dict[str, Any]], cle: str) -> float:
    return sum(r[cle] for r in scores) / len(scores) if scores else 0.0


def run(
    golden_path: str,
    ingestion_config_path: str,
    report_path: str,
    w_dense: float = DEFAULT_W_DENSE,
    w_sparse: float = DEFAULT_W_SPARSE,
) -> dict[str, Any]:
    questions = load_golden_dataset(golden_path)
    print(f"[INFO] {len(questions)} questions chargées depuis {golden_path}")

    print("[INFO] Reconstruction du corpus de référence (identique à l'ingestion)…")
    texts, ingestion_config = build_reference_corpus(ingestion_config_path)
    bm25, vocab = build_bm25_index(texts)

    print(f"[INFO] Chargement du modèle : {ingestion_config['modele_embedding']}")
    model = SentenceTransformer(ingestion_config["modele_embedding"])

    settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {settings.qdrant_url}")
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    collection_name = ingestion_config["collection_name"]

    results = []
    for question in questions:
        result = evaluate_question(
            question, model, bm25, vocab, client, collection_name,
            w_dense=w_dense, w_sparse=w_sparse,
        )
        results.append(result)

    enrichir_echecs_avec_payload(client, collection_name, results)

    scores = [r for r in results if not r["reference_invalide"]]
    exclus = [r for r in results if r["reference_invalide"]]

    report = {
        "w_dense": w_dense,
        "w_sparse": w_sparse,
        "nb_questions": len(results),
        "nb_questions_scorees": len(scores),
        "nb_questions_exclues": len(exclus),
        "recall_at_1_moyen": _moyenne(scores, "recall_at_1"),
        "recall_at_3_moyen": _moyenne(scores, "recall_at_3"),
        "recall_at_5_moyen": _moyenne(scores, "recall_at_5"),
        "recall_at_10_moyen": _moyenne(scores, "recall_at_10"),
        "precision_at_1_moyen": _moyenne(scores, "precision_at_1"),
        "mrr": _moyenne(scores, "reciprocal_rank"),
        "details": results,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(report)
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(f"\n[RESUME EVALUATION RETRIEVAL]")
    print(f"  Questions chargées                     : {report['nb_questions']}")
    print(f"  Questions scorées                       : {report['nb_questions_scorees']}")
    print(f"  Questions exclues (référence invalide)  : {report['nb_questions_exclues']}")
    print()
    print(f"  Recall@1    : {report['recall_at_1_moyen']:.2%}")
    print(f"  Recall@3    : {report['recall_at_3_moyen']:.2%}")
    print(f"  Recall@5    : {report['recall_at_5_moyen']:.2%}")
    print(f"  Recall@10   : {report['recall_at_10_moyen']:.2%}")
    print(f"  Precision@1 : {report['precision_at_1_moyen']:.2%}")
    print(f"  MRR         : {report['mrr']:.4f}")

    details = report["details"]

    exclus = [r for r in details if r.get("reference_invalide")]
    if exclus:
        print(f"\n  Questions exclues (id_reference introuvable dans Qdrant) :")
        for r in exclus:
            print(f"    - {r['id_test']} | categorie={r['categorie']}")

    scores = [r for r in details if not r.get("reference_invalide")]
    print(f"\n  Détail par question :")
    print(f"    {'id_test':<20} {'categorie':<22} {'rang':>5}  {'@1':>4} {'@3':>4} {'@5':>4} {'@10':>4}")
    for r in scores:
        rang = r["rang_trouve"] if r["rang_trouve"] is not None else "-"

        def marque(profondeur: int) -> str:
            return "PASS" if r[f"recall_at_{profondeur}"] == 1 else "FAIL"

        print(
            f"    {r['id_test']:<20} {r['categorie']:<22} {str(rang):>5}  "
            f"{marque(1):>4} {marque(3):>4} {marque(5):>4} {marque(10):>4}"
        )

    echecs = [r for r in scores if r["recall_at_3"] == 0]
    if echecs:
        print(f"\n  Inspection des échecs (recall@3=0), vérité-terrain vs top-{ECHEC_INSPECTION_TOP_N} retourné :")
        for r in echecs:
            print(f"\n    {r['id_test']} | categorie={r['categorie']}")
            vt = r.get("verite_terrain_resume")
            if vt:
                print(f"      Attendu  : [{vt['categorie']}] {vt['question']} → {vt['reponse']}")
            for i, top in enumerate(r.get("top_resultats_resume", []), start=1):
                if top:
                    print(f"      Retourné #{i} : [{top['categorie']}] {top['question']} → {top['reponse']}")
    else:
        print(f"\n  Aucune question en échec (recall@3=1 pour toutes les questions scorées).")
    print()


# ── Grid search du poids RRF (étape 1 — coût zéro, pas d'appel génératif) ────

def grid_search_rrf_weights(
    golden_path: str,
    ingestion_config_path: str,
    poids_dense_testes: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Recherche en grille du poids RRF (w_dense variable, w_sparse = 1 - w_dense).

    Le pool de candidats dense/sparse (search_batch) est récupéré UNE SEULE FOIS
    par question — indépendant du poids de fusion — puis refusionné en mémoire
    (reciprocal_rank_fusion) pour chaque combinaison testée. Aucun appel réseau
    supplémentaire par poids : seule la fusion change, jamais le pool candidat.

    Retourne une liste de rapports agrégés (un par poids testé), chacun avec le
    même format de moyennes que run() (recall@k, precision@1, MRR) + le détail
    par question pour permettre l'inspection groupe par groupe.
    """
    poids_dense_testes = poids_dense_testes or DEFAULT_GRID_W_DENSE

    questions = load_golden_dataset(golden_path)
    print(f"[INFO] {len(questions)} questions chargées depuis {golden_path}")

    print("[INFO] Reconstruction du corpus de référence (identique à l'ingestion)…")
    texts, ingestion_config = build_reference_corpus(ingestion_config_path)
    bm25, vocab = build_bm25_index(texts)

    print(f"[INFO] Chargement du modèle : {ingestion_config['modele_embedding']}")
    model = SentenceTransformer(ingestion_config["modele_embedding"])

    settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {settings.qdrant_url}")
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    collection_name = ingestion_config["collection_name"]

    candidats_par_question = []
    for question in questions:
        ground_truth_id = build_point_id(question["id_reference"], question["source_type"])
        existants = client.retrieve(collection_name=collection_name, ids=[ground_truth_id])
        if not existants:
            candidats_par_question.append({"reference_invalide": True, "question": question})
            continue

        query_text = question["question_client"]
        dense_query = model.encode(
            query_text, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        sparse_query = vectorize_bm25(tokenize(query_text), bm25, vocab)
        dense_results, sparse_results = fetch_dense_sparse_candidates(
            client, collection_name, dense_query, sparse_query
        )

        candidats_par_question.append({
            "reference_invalide": False,
            "question": question,
            "ground_truth_id": ground_truth_id,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
        })

    resultats_grille = []
    for w_dense in poids_dense_testes:
        w_sparse = round(1.0 - w_dense, 2)
        details = []
        for candidat in candidats_par_question:
            if candidat["reference_invalide"]:
                continue

            fused = reciprocal_rank_fusion(
                candidat["dense_results"], candidat["sparse_results"],
                w_dense=w_dense, w_sparse=w_sparse,
            )
            top_ids = [point_id for point_id, _ in fused[:TOP_K_MAX]]
            ground_truth_id = candidat["ground_truth_id"]
            rang_trouve = top_ids.index(ground_truth_id) + 1 if ground_truth_id in top_ids else None
            recalls = {
                f"recall_at_{profondeur}": 1 if (rang_trouve is not None and rang_trouve <= profondeur) else 0
                for profondeur in RECALL_DEPTHS
            }
            precision_1 = 1 if top_ids and top_ids[0] == ground_truth_id else 0
            reciprocal_rank = 1.0 / rang_trouve if rang_trouve is not None else 0.0

            details.append({
                "id_test": candidat["question"]["id_test"],
                "categorie": candidat["question"]["categorie"],
                "rang_trouve": rang_trouve,
                **recalls,
                "precision_at_1": precision_1,
                "reciprocal_rank": reciprocal_rank,
            })

        resultats_grille.append({
            "w_dense": w_dense,
            "w_sparse": w_sparse,
            "nb_questions_scorees": len(details),
            "recall_at_1_moyen": _moyenne(details, "recall_at_1"),
            "recall_at_3_moyen": _moyenne(details, "recall_at_3"),
            "recall_at_5_moyen": _moyenne(details, "recall_at_5"),
            "recall_at_10_moyen": _moyenne(details, "recall_at_10"),
            "precision_at_1_moyen": _moyenne(details, "precision_at_1"),
            "mrr": _moyenne(details, "reciprocal_rank"),
            "details": details,
        })

    return resultats_grille


# ── Évaluation dense-only (sans fusion RRF ni composante BM25) ──────────────
# Objectif : isoler l'effet de la fusion sparse sur le retrieval. Un mode
# séparé (fichier de rapport distinct, pas de flag par défaut) — ne modifie
# jamais le comportement de hybrid_search()/run() ni retriever.py (production).

DEFAULT_DENSE_ONLY_REPORT_PATH = "tests/retrieval_eval_dense_only_report.json"

# Questions jamais retrouvées en hybride (rang_trouve=null, absentes du top-10
# fusionné) d'après tests/retrieval_eval_report.json — cf. évaluation du
# 2026-07-20. Sert de point de comparaison ciblé pour le dense-only.
GROUPE_B_ID_TESTS = [
    "gd_facturation_005",  # Conventions
    "gd_facturation_006",  # Devis
    "gd_facturation_011",  # Forfaits
    "gd_facturation_015",  # Paramétrage
]
GROUPE_B_INSPECTION_LIMIT = 20  # profondeur de recherche dédiée à cette inspection


def dense_only_search(
    client: QdrantClient,
    collection_name: str,
    dense_query: list[float],
    limit: int = TOP_K_MAX,
) -> list[tuple[str, float]]:
    """
    Recherche dense pure (client.search sur le vecteur nommé "dense"), SANS
    fusion RRF ni composante sparse BM25 — sert à isoler l'effet de la fusion
    hybride (comparaison avec hybrid_search()).
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name=DENSE_VECTOR_NAME, vector=dense_query),
        limit=limit,
        with_payload=False,
    )
    return [(str(hit.id), hit.score) for hit in results]


def evaluate_question_dense_only(
    question: dict[str, Any],
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
) -> dict[str, Any]:
    """
    Variante dense-only de evaluate_question() : mêmes métriques et même
    garde-fou vérité-terrain, mais classement obtenu uniquement par similarité
    cosine dense (dense_only_search), jamais par fusion RRF/BM25.
    """
    ground_truth_id = build_point_id(question["id_reference"], question["source_type"])

    existants = client.retrieve(collection_name=collection_name, ids=[ground_truth_id])
    if not existants:
        return {
            "id_test": question["id_test"],
            "categorie": question["categorie"],
            "interface_ou_serveur": question["interface_ou_serveur"],
            "reference_invalide": True,
        }

    query_text = question["question_client"]
    dense_query = model.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    top_results = dense_only_search(client, collection_name, dense_query, limit=TOP_K_MAX)
    top_ids = [point_id for point_id, _ in top_results]

    trouve = ground_truth_id in top_ids
    rang_trouve = top_ids.index(ground_truth_id) + 1 if trouve else None
    precision_1 = 1 if top_ids and top_ids[0] == ground_truth_id else 0
    score_premier = top_results[0][1] if top_results else None
    reciprocal_rank = 1.0 / rang_trouve if rang_trouve is not None else 0.0

    recalls = {
        f"recall_at_{profondeur}": 1 if (rang_trouve is not None and rang_trouve <= profondeur) else 0
        for profondeur in RECALL_DEPTHS
    }

    return {
        "id_test": question["id_test"],
        "categorie": question["categorie"],
        "interface_ou_serveur": question["interface_ou_serveur"],
        "reference_invalide": False,
        "trouve_top3": recalls["recall_at_3"] == 1,
        "rang_trouve": rang_trouve,
        **recalls,
        "precision_at_1": precision_1,
        "reciprocal_rank": reciprocal_rank,
        "score_du_premier_resultat": score_premier,
        "ground_truth_id": ground_truth_id,
        "top_ids": top_ids,
    }


def run_dense_only(
    golden_path: str,
    ingestion_config_path: str,
    report_path: str = DEFAULT_DENSE_ONLY_REPORT_PATH,
) -> dict[str, Any]:
    """
    Évaluation dense-only sur le golden dataset complet (mêmes 20 questions,
    même collection que run()) : recall@1/3/5/10, precision@1, MRR calculés
    uniquement à partir du classement dense (aucune fusion RRF/BM25). Écrit
    dans un fichier de rapport séparé — ne touche jamais
    tests/retrieval_eval_report.json (rapport hybride de production).
    """
    questions = load_golden_dataset(golden_path)
    print(f"[INFO] {len(questions)} questions chargées depuis {golden_path}")

    ingestion_config = load_ingestion_config(ingestion_config_path)
    print(f"[INFO] Chargement du modèle : {ingestion_config['modele_embedding']}")
    model = SentenceTransformer(ingestion_config["modele_embedding"])

    settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {settings.qdrant_url}")
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    collection_name = ingestion_config["collection_name"]

    results = [
        evaluate_question_dense_only(question, model, client, collection_name)
        for question in questions
    ]

    scores = [r for r in results if not r["reference_invalide"]]
    exclus = [r for r in results if r["reference_invalide"]]

    report = {
        "mode": "dense_only",
        "nb_questions": len(results),
        "nb_questions_scorees": len(scores),
        "nb_questions_exclues": len(exclus),
        "recall_at_1_moyen": _moyenne(scores, "recall_at_1"),
        "recall_at_3_moyen": _moyenne(scores, "recall_at_3"),
        "recall_at_5_moyen": _moyenne(scores, "recall_at_5"),
        "recall_at_10_moyen": _moyenne(scores, "recall_at_10"),
        "precision_at_1_moyen": _moyenne(scores, "precision_at_1"),
        "mrr": _moyenne(scores, "reciprocal_rank"),
        "details": results,
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(report)
    return report


def print_comparaison_hybride_vs_dense(
    hybride: dict[str, Any] | None, dense: dict[str, Any]
) -> None:
    """Tableau comparatif hybride vs dense-only, métrique par métrique."""
    print("\n[COMPARAISON HYBRIDE vs DENSE-ONLY]")
    if hybride is None:
        print("  (rapport hybride introuvable — comparaison ignorée, dense-only seul affiché ci-dessus)")
        return

    metriques = [
        ("Recall@1", "recall_at_1_moyen"),
        ("Recall@3", "recall_at_3_moyen"),
        ("Recall@5", "recall_at_5_moyen"),
        ("Recall@10", "recall_at_10_moyen"),
        ("Precision@1", "precision_at_1_moyen"),
        ("MRR", "mrr"),
    ]
    print(f"  {'Métrique':<12} {'Hybride':>10} {'Dense-only':>12} {'Écart':>10}")
    for label, cle in metriques:
        v_hybride = hybride.get(cle, 0.0)
        v_dense = dense.get(cle, 0.0)
        ecart = v_dense - v_hybride
        if cle == "mrr":
            print(f"  {label:<12} {v_hybride:>10.4f} {v_dense:>12.4f} {ecart:>+10.4f}")
        else:
            print(f"  {label:<12} {v_hybride:>9.1%} {v_dense:>11.1%} {ecart:>+9.1%}")
    print()


def inspect_groupe_b_dense_only(
    golden_path: str,
    ingestion_config_path: str,
    id_tests: list[str] = GROUPE_B_ID_TESTS,
    limit: int = GROUPE_B_INSPECTION_LIMIT,
) -> list[dict[str, Any]]:
    """
    Inspection ciblée des questions jamais retrouvées en hybride (GROUPE_B) :
    relance une recherche dense pure à profondeur `limit` (20, plus profond que
    TOP_K_MAX=10 utilisé pour les métriques standard) pour déterminer si la
    fusion sparse les faisait sortir du pool (rang dense-only <= 10 → effet de
    bord de la fusion) ou si l'embedding dense lui-même échoue (absentes du
    top-20 → vrai problème de couverture vocabulaire/domaine).
    """
    questions = {q["id_test"]: q for q in load_golden_dataset(golden_path)}
    ingestion_config = load_ingestion_config(ingestion_config_path)
    model = SentenceTransformer(ingestion_config["modele_embedding"])
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    collection_name = ingestion_config["collection_name"]

    resultats = []
    print(f"\n[INSPECTION GROUPE B — jamais trouvées en hybride, dense-only profondeur {limit}]")
    for id_test in id_tests:
        question = questions.get(id_test)
        if question is None:
            print(f"  {id_test} : introuvable dans le golden dataset")
            continue

        ground_truth_id = build_point_id(question["id_reference"], question["source_type"])
        existants = client.retrieve(collection_name=collection_name, ids=[ground_truth_id])
        if not existants:
            print(f"  {id_test} | categorie={question['categorie']} : référence invalide (exclue)")
            continue

        query_text = question["question_client"]
        dense_query = model.encode(
            query_text, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        top_results = dense_only_search(client, collection_name, dense_query, limit=limit)
        top_ids = [point_id for point_id, _ in top_results]
        rang = top_ids.index(ground_truth_id) + 1 if ground_truth_id in top_ids else None

        diagnostic = (
            f"rang {rang} en dense-only"
            if rang is not None
            else f"absent du top-{limit}"
        )
        print(f"  {id_test} | categorie={question['categorie']:<15} : {diagnostic}")

        resultats.append({
            "id_test": id_test,
            "categorie": question["categorie"],
            "rang_dense_only": rang,
            "absent_du_top": rang is None,
        })

    print()
    return resultats


# ── Reranking cross-encoder (2e étage, ai_engine/rag/reranker.py) ────────────
# Réutilise le pool hybride de premier étage (fetch_dense_sparse_candidates +
# reciprocal_rank_fusion) — pas de duplication de la fusion RRF ni des métriques
# de base. Le "rang_avant" (baseline de comparaison) est calculé sur CE MÊME
# pool à profondeur first_stage_k (pas TOP_K_MAX=10) pour isoler l'effet du
# rerank de celui d'un pool plus profond.

DEFAULT_FIRST_STAGE_K = 20
DEFAULT_TOP_N = 3
DEFAULT_RERANK_REPORT_CSV_PATH = "tests/rerank_score_report.csv"


def evaluate_question_rerank(
    question: dict[str, Any],
    model: SentenceTransformer,
    bm25,
    vocab: dict[str, int],
    client: QdrantClient,
    collection_name: str,
    reranker,
    first_stage_k: int = DEFAULT_FIRST_STAGE_K,
) -> dict[str, Any]:
    """
    Évalue une question du golden dataset après reranking cross-encoder du pool
    hybride de premier étage (profondeur first_stage_k).

    Le pool est refusionné en mémoire (mêmes fonctions que hybrid_search), puis
    rerankage intégral (top_n=len(pool)) pour obtenir un classement complet
    permettant de calculer recall@1/3/5 et MRR sur la liste rerankée, comme
    pour evaluate_question(). Le "rang_avant" (avant rerank) sert de baseline
    de comparaison à profondeur first_stage_k (cf. note de section ci-dessus).
    """
    ground_truth_id = build_point_id(question["id_reference"], question["source_type"])

    existants = client.retrieve(collection_name=collection_name, ids=[ground_truth_id])
    if not existants:
        return {
            "id_test": question["id_test"],
            "categorie": question["categorie"],
            "reference_invalide": True,
        }

    query_text = question["question_client"]
    dense_query = model.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    sparse_query = vectorize_bm25(tokenize(query_text), bm25, vocab)

    dense_results, sparse_results = fetch_dense_sparse_candidates(
        client, collection_name, dense_query, sparse_query, limit=first_stage_k
    )
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    pool_ids = [point_id for point_id, _ in fused[:first_stage_k]]

    in_pool = ground_truth_id in pool_ids
    rang_avant = pool_ids.index(ground_truth_id) + 1 if in_pool else None

    if not pool_ids:
        return {
            "id_test": question["id_test"],
            "categorie": question["categorie"],
            "reference_invalide": False,
            "in_pool": False,
            "rang_avant": None,
            "rang_apres": None,
            "reranker_score_bonne_fiche": None,
            "top1_reranker_score": None,
            "recall_at_1": 0, "recall_at_3": 0, "recall_at_5": 0,
            "precision_at_1": 0, "reciprocal_rank": 0.0,
        }

    points = client.retrieve(collection_name=collection_name, ids=pool_ids, with_payload=True)
    payload_par_id = {str(p.id): p.payload for p in points}
    candidates = [
        {"id": point_id, "payload": payload_par_id[point_id]}
        for point_id in pool_ids
        if point_id in payload_par_id
    ]

    reranked = reranker.rerank(query_text, candidates, top_n=len(candidates))
    reranked_ids = [c["id"] for c in reranked]

    rang_apres = reranked_ids.index(ground_truth_id) + 1 if in_pool and ground_truth_id in reranked_ids else None
    reranker_score_bonne_fiche = (
        next((c["reranker_score"] for c in reranked if c["id"] == ground_truth_id), None)
        if in_pool else None
    )
    top1_reranker_score = reranked[0]["reranker_score"] if reranked else None
    reranked_top1_id = reranked[0]["id"] if reranked else None

    recalls = {
        f"recall_at_{profondeur}": 1 if (rang_apres is not None and rang_apres <= profondeur) else 0
        for profondeur in [1, 3, 5]
    }
    precision_1 = 1 if reranked_ids and reranked_ids[0] == ground_truth_id else 0
    reciprocal_rank = 1.0 / rang_apres if rang_apres is not None else 0.0

    return {
        "id_test": question["id_test"],
        "categorie": question["categorie"],
        "reference_invalide": False,
        "in_pool": in_pool,
        "rang_avant": rang_avant,
        "rang_apres": rang_apres,
        "reranker_score_bonne_fiche": reranker_score_bonne_fiche,
        "top1_reranker_score": top1_reranker_score,
        "reranked_top1_id": reranked_top1_id,
        **recalls,
        "precision_at_1": precision_1,
        "reciprocal_rank": reciprocal_rank,
    }


# ── Questions négatives (id_reference=null, gate d'abstention) ──────────────
# Aucune vérité-terrain à retrouver : le pool hybride + rerank est calculé de
# la même façon que pour un positif (réutilise fetch_dense_sparse_candidates +
# reciprocal_rank_fusion), mais seul le score du 1er résultat après rerank
# importe — c'est ce score qui, comparé à un seuil d'abstention, déterminerait
# si le système répond à tort à une question hors-scope.

# Seuil du gate d'abstention : lu depuis config.py (RAG_CONFIDENCE_THRESHOLD,
# 0.60 par défaut — validé sur golden_dataset_v3.json, cf. commentaire config.py).
# Les 2 seuils de comparaison (+/- 0.10) s'adaptent automatiquement si la valeur
# change en .env, plutôt que de figer 0.50/0.60/0.70 en dur.
RAG_CONFIDENCE_THRESHOLD = get_settings().rag_confidence_threshold
NEGATIFS_SEUILS_ABSTENTION = (
    round(RAG_CONFIDENCE_THRESHOLD - 0.10, 2),
    RAG_CONFIDENCE_THRESHOLD,
    round(RAG_CONFIDENCE_THRESHOLD + 0.10, 2),
)


def _type_effectif(question: dict[str, Any]) -> str:
    """
    Type déclaré de la question, ou valeur par défaut pour les entrées legacy
    (les 20 questions reprises de golden_dataset_v1.json n'ont pas de champ
    "type" — ce sont des tickets réels, donc "positif_tickets_reels" par défaut).
    """
    if question.get("type"):
        return question["type"]
    return "positif_tickets_reels" if question.get("id_reference") is not None else "negatif_tickets_reels"


def evaluate_negative_question(
    question: dict[str, Any],
    model: SentenceTransformer,
    bm25,
    vocab: dict[str, int],
    client: QdrantClient,
    collection_name: str,
    reranker,
    first_stage_k: int = DEFAULT_FIRST_STAGE_K,
) -> dict[str, Any]:
    """
    Évalue une question négative (id_reference=null, réponse attendue ESCALADE) :
    calcule le pool hybride puis reranke, et retient uniquement le
    top1_reranker_score (aucune vérité-terrain à situer dans le classement).
    """
    query_text = question["question_client"]
    dense_query = model.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    sparse_query = vectorize_bm25(tokenize(query_text), bm25, vocab)

    dense_results, sparse_results = fetch_dense_sparse_candidates(
        client, collection_name, dense_query, sparse_query, limit=first_stage_k
    )
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    pool_ids = [point_id for point_id, _ in fused[:first_stage_k]]

    top1_reranker_score = None
    if pool_ids:
        points = client.retrieve(collection_name=collection_name, ids=pool_ids, with_payload=True)
        payload_par_id = {str(p.id): p.payload for p in points}
        candidates = [
            {"id": point_id, "payload": payload_par_id[point_id]}
            for point_id in pool_ids
            if point_id in payload_par_id
        ]
        reranked = reranker.rerank(query_text, candidates, top_n=1)
        top1_reranker_score = reranked[0]["reranker_score"] if reranked else None

    return {
        "id_test": question["id_test"],
        "categorie": question["categorie"],
        "type": _type_effectif(question),
        "question_client": query_text,
        "top1_reranker_score": top1_reranker_score,
    }


def _percentile(valeurs_triees: list[float], pct: float) -> float | None:
    if not valeurs_triees:
        return None
    n = len(valeurs_triees)
    if n == 1:
        return valeurs_triees[0]
    idx = pct * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return valeurs_triees[lo] + (valeurs_triees[hi] - valeurs_triees[lo]) * frac


def compute_negatifs_stats(
    negatifs_results: list[dict[str, Any]],
    seuils: tuple[float, ...] = NEGATIFS_SEUILS_ABSTENTION,
) -> dict[str, Any]:
    """
    Statistiques du gate d'abstention sur les négatifs : distribution du
    top1_reranker_score et taux de faux positifs (négatifs qui dépasseraient
    un seuil d'abstention donné et recevraient donc une réponse à tort).
    """
    nb_negatifs = len(negatifs_results)
    scores = [
        r["top1_reranker_score"] for r in negatifs_results if r["top1_reranker_score"] is not None
    ]
    scores_tries = sorted(scores)

    taux_faux_positifs = {
        seuil: (sum(1 for s in scores if s > seuil) / nb_negatifs if nb_negatifs else 0.0)
        for seuil in seuils
    }

    return {
        "nb_negatifs": nb_negatifs,
        "nb_scores_valides": len(scores),
        "top1_score_min": scores_tries[0] if scores_tries else None,
        "top1_score_p25": _percentile(scores_tries, 0.25),
        "top1_score_median": _percentile(scores_tries, 0.5),
        "top1_score_p75": _percentile(scores_tries, 0.75),
        "top1_score_max": scores_tries[-1] if scores_tries else None,
        "top1_score_moyenne": (sum(scores) / len(scores)) if scores else None,
        "taux_faux_positifs": taux_faux_positifs,
    }


def _baseline_recalls_depuis_rang(rang: int | None) -> dict[str, int]:
    return {
        f"recall_at_{profondeur}": 1 if (rang is not None and rang <= profondeur) else 0
        for profondeur in [1, 3, 5]
    }


def run_rerank(
    golden_path: str,
    ingestion_config_path: str,
    first_stage_k: int = DEFAULT_FIRST_STAGE_K,
    top_n: int = DEFAULT_TOP_N,
    report_csv_path: str = DEFAULT_RERANK_REPORT_CSV_PATH,
    dump_top1_positifs: bool = False,
    top1_positifs_csv_path: str = "tests/top1_positifs_dump.csv",
) -> dict[str, Any]:
    """
    Évaluation avec reranking cross-encoder (2e étage), sur le golden dataset
    complet. Ne modifie ni retriever.py ni la config de production ; lecture
    Qdrant seule (client.retrieve/search_batch).
    """
    from ai_engine.rag.reranker import Reranker

    questions = load_golden_dataset(golden_path)
    print(f"[INFO] {len(questions)} questions chargées depuis {golden_path}")

    # Les négatifs (id_reference=null, réponse attendue ESCALADE) n'ont pas de
    # vérité-terrain à situer dans un classement : ils sont exclus du calcul
    # de recall/precision/MRR et évalués séparément (gate d'abstention).
    positifs = [q for q in questions if q.get("id_reference") is not None]
    negatifs = [q for q in questions if q.get("id_reference") is None]

    print("[INFO] Reconstruction du corpus de référence (identique à l'ingestion)…")
    texts, ingestion_config = build_reference_corpus(ingestion_config_path)
    bm25, vocab = build_bm25_index(texts)

    print(f"[INFO] Chargement du modèle dense : {ingestion_config['modele_embedding']}")
    model = SentenceTransformer(ingestion_config["modele_embedding"])

    print(f"[INFO] Chargement du reranker cross-encoder (BAAI/bge-reranker-v2-m3)…")
    reranker = Reranker()

    settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {settings.qdrant_url}")
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    collection_name = ingestion_config["collection_name"]

    results = [
        evaluate_question_rerank(
            question, model, bm25, vocab, client, collection_name, reranker,
            first_stage_k=first_stage_k,
        )
        for question in positifs
    ]
    id_test_to_type = {question["id_test"]: _type_effectif(question) for question in positifs}

    negatifs_results = [
        evaluate_negative_question(
            question, model, bm25, vocab, client, collection_name, reranker,
            first_stage_k=first_stage_k,
        )
        for question in negatifs
    ]

    scores = [r for r in results if not r["reference_invalide"]]
    exclus = [r for r in results if r["reference_invalide"]]

    # Baseline "avant rerank" recalculée sur rang_avant (même profondeur first_stage_k),
    # pour comparer pool identique avant/après plutôt que pool hybride TOP_K_MAX=10.
    baseline_details = [
        {**_baseline_recalls_depuis_rang(r["rang_avant"]),
         "precision_at_1": 1 if r["rang_avant"] == 1 else 0,
         "reciprocal_rank": 1.0 / r["rang_avant"] if r["rang_avant"] is not None else 0.0}
        for r in scores
    ]

    report = {
        "first_stage_k": first_stage_k,
        "top_n": top_n,
        "nb_questions": len(positifs),
        "nb_questions_scorees": len(scores),
        "nb_questions_exclues": len(exclus),
        "baseline_avant_rerank": {
            "recall_at_1_moyen": _moyenne(baseline_details, "recall_at_1"),
            "recall_at_3_moyen": _moyenne(baseline_details, "recall_at_3"),
            "recall_at_5_moyen": _moyenne(baseline_details, "recall_at_5"),
            "precision_at_1_moyen": _moyenne(baseline_details, "precision_at_1"),
            "mrr": _moyenne(baseline_details, "reciprocal_rank"),
        },
        "apres_rerank": {
            "recall_at_1_moyen": _moyenne(scores, "recall_at_1"),
            "recall_at_3_moyen": _moyenne(scores, "recall_at_3"),
            "recall_at_5_moyen": _moyenne(scores, "recall_at_5"),
            "precision_at_1_moyen": _moyenne(scores, "precision_at_1"),
            "mrr": _moyenne(scores, "reciprocal_rank"),
        },
        "details": results,
        "id_test_to_type": id_test_to_type,
        "negatifs": {
            "stats": compute_negatifs_stats(negatifs_results),
            "details": negatifs_results,
        },
    }

    _write_rerank_csv(results, id_test_to_type, negatifs_results, report_csv_path)
    if dump_top1_positifs:
        _dump_top1_positifs(golden_path, results, id_test_to_type, top1_positifs_csv_path)
    print_rerank_summary(report)
    print_negatifs_summary(report["negatifs"]["stats"], negatifs_results)
    return report


def _write_rerank_csv(
    results: list[dict[str, Any]],
    id_test_to_type: dict[str, str],
    negatifs_results: list[dict[str, Any]],
    report_csv_path: str,
) -> None:
    import csv

    Path(report_csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id_test", "categorie", "rang_avant", "rang_apres",
        "reranker_score_bonne_fiche", "top1_reranker_score", "in_pool", "type",
    ]
    with open(report_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if r.get("reference_invalide"):
                continue
            row = dict(r)
            row["rang_avant"] = row["rang_avant"] if row["rang_avant"] is not None else "absent"
            row["rang_apres"] = row["rang_apres"] if row["rang_apres"] is not None else "absent"
            row["type"] = id_test_to_type.get(row["id_test"])
            writer.writerow(row)

        for r in negatifs_results:
            writer.writerow({
                "id_test": r["id_test"],
                "categorie": r["categorie"],
                "rang_avant": "N/A",
                "rang_apres": "N/A",
                "reranker_score_bonne_fiche": None,
                "top1_reranker_score": r["top1_reranker_score"],
                "in_pool": False,
                "type": r["type"],
            })


def _dump_top1_positifs(
    golden_path: str,
    results: list[dict[str, Any]],
    id_test_to_type: dict[str, str],
    csv_path: str,
) -> None:
    """
    Export des résultats rang-1 pour questions positives (et résumé chiffré).
    Colonnes : question_id, id_reference_attendu, id_reference_rang1,
    rang1_correct, top1_score, type.
    """
    import csv

    questions_data = load_golden_dataset(golden_path)
    id_test_to_question = {q["id_test"]: q for q in questions_data}

    rows = []
    for result in results:
        if result.get("reference_invalide"):
            continue

        id_test = result["id_test"]
        question = id_test_to_question.get(id_test)
        if not question:
            continue

        id_reference_attendu = build_point_id(question["id_reference"], question["source_type"])
        id_reference_rang1 = (
            result.get("reranked_top1_id")
            if "reranked_top1_id" in result
            else None
        )
        rang1_correct = (
            id_reference_rang1 is not None and
            id_reference_rang1 == id_reference_attendu
        )
        top1_score = result.get("top1_reranker_score")
        type_val = id_test_to_type.get(id_test)

        rows.append({
            "question_id": id_test,
            "id_reference_attendu": id_reference_attendu,
            "id_reference_rang1": id_reference_rang1,
            "rang1_correct": rang1_correct,
            "top1_score": top1_score,
            "type": type_val,
        })

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_id", "id_reference_attendu", "id_reference_rang1", "rang1_correct", "top1_score", "type"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _print_top1_positifs_summary(rows)


def _print_top1_positifs_summary(rows: list[dict[str, Any]]) -> None:
    """Résumé chiffré des résultats rang-1 par groupe (correct/incorrect)."""

    def _percentiles(values: list[float]) -> tuple[float, float, float]:
        """Retourne (min, médiane, max)."""
        if not values:
            return None, None, None
        s = sorted(values)
        n = len(s)
        min_v = s[0]
        max_v = s[-1]
        mid = n // 2
        median = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2
        return min_v, median, max_v

    corrects = [r for r in rows if r["rang1_correct"]]
    incorrects = [r for r in rows if not r["rang1_correct"]]

    correct_scores = [r["top1_score"] for r in corrects if r["top1_score"] is not None]
    incorrect_scores = [r["top1_score"] for r in incorrects if r["top1_score"] is not None]

    min_c, med_c, max_c = _percentiles(correct_scores)
    min_i, med_i, max_i = _percentiles(incorrect_scores)

    print(f"\n[DUMP TOP1 POSITIFS]")
    print(f"  Groupe rang-1 CORRECT :")
    print(f"    n={len(corrects)}, min={min_c:.3f}, médiane={med_c:.3f}, max={max_c:.3f}" if min_c is not None else f"    n=0")

    print(f"  Groupe rang-1 INCORRECT :")
    print(f"    n={len(incorrects)}, min={min_i:.3f}, médiane={med_i:.3f}, max={max_i:.3f}" if min_i is not None else f"    n=0")

    critical = sum(1 for r in incorrects if r["top1_score"] is not None and r["top1_score"] >= 0.80)
    print(f"  Positifs à rang-1 incorrect avec top1_score >= 0.80 : {critical}")
    print()


def print_breakdown_positifs_par_type(
    details: list[dict[str, Any]], id_test_to_type: dict[str, str]
) -> None:
    """
    Recall@3 et precision@1 après rerank, groupés par "type" (positif_tickets_reels
    vs positif_synthetique) — permet de voir si les questions synthétiques se
    comportent différemment des tickets réels sans rejouer le calcul de base.
    """
    scores = [r for r in details if not r.get("reference_invalide")]
    types = sorted({id_test_to_type[r["id_test"]] for r in scores if r["id_test"] in id_test_to_type})

    print(f"\n=== BREAKDOWN PAR TYPE (positifs uniquement) ===")
    print(f"  {'Type':<24}| {'N':>3} | {'Recall@3':>8} | {'Precision@1':>11}")
    for type_val in types:
        subset = [r for r in scores if id_test_to_type.get(r["id_test"]) == type_val]
        recall3 = _moyenne(subset, "recall_at_3")
        precision1 = _moyenne(subset, "precision_at_1")
        print(f"  {type_val:<24}| {len(subset):>3} | {recall3:>7.1%} | {precision1:>10.1%}")


def print_rerank_summary(report: dict[str, Any]) -> None:
    print(f"\n[RESUME EVALUATION RERANK — first_stage_k={report['first_stage_k']}, top_n={report['top_n']}]")
    print(f"  Questions chargées                     : {report['nb_questions']}")
    print(f"  Questions scorées                       : {report['nb_questions_scorees']}")
    print(f"  Questions exclues (référence invalide)  : {report['nb_questions_exclues']}")

    baseline = report["baseline_avant_rerank"]
    apres = report["apres_rerank"]
    metriques = [
        ("Recall@1", "recall_at_1_moyen"),
        ("Recall@3", "recall_at_3_moyen"),
        ("Recall@5", "recall_at_5_moyen"),
        ("Precision@1", "precision_at_1_moyen"),
        ("MRR", "mrr"),
    ]
    print(f"\n  {'Métrique':<12} {'Avant rerank':>14} {'Après rerank':>14} {'Écart':>10}")
    for label, cle in metriques:
        v_avant = baseline[cle]
        v_apres = apres[cle]
        ecart = v_apres - v_avant
        if cle == "mrr":
            print(f"  {label:<12} {v_avant:>14.4f} {v_apres:>14.4f} {ecart:>+10.4f}")
        else:
            print(f"  {label:<12} {v_avant:>13.1%} {v_apres:>13.1%} {ecart:>+9.1%}")

    print_breakdown_positifs_par_type(report["details"], report.get("id_test_to_type", {}))

    print(f"\n  Détail par question :")
    print(f"    {'id_test':<20} {'categorie':<22} {'in_pool':>8} {'rang_avant':>11} {'rang_apres':>11} {'score_bonne':>12} {'top1_score':>11}")
    for r in report["details"]:
        if r.get("reference_invalide"):
            continue
        rang_avant = r["rang_avant"] if r["rang_avant"] is not None else "-"
        rang_apres = r["rang_apres"] if r["rang_apres"] is not None else "-"
        score_bonne = f"{r['reranker_score_bonne_fiche']:.3f}" if r["reranker_score_bonne_fiche"] is not None else "-"
        top1_score = f"{r['top1_reranker_score']:.3f}" if r["top1_reranker_score"] is not None else "-"
        print(
            f"    {r['id_test']:<20} {r['categorie']:<22} {str(r['in_pool']):>8} "
            f"{str(rang_avant):>11} {str(rang_apres):>11} {score_bonne:>12} {top1_score:>11}"
        )

    # Résumé séparation des scores : bonne fiche en rang 1 (in-pool) vs top1_score
    # des questions dont l'id_reference est absente du pool (cas structurels type
    # Devis/Forfaits) — pour visualiser si un seuil d'abstention sépare les deux.
    scores_rang1 = [
        r["reranker_score_bonne_fiche"] for r in report["details"]
        if not r.get("reference_invalide") and r["rang_apres"] == 1
        and r["reranker_score_bonne_fiche"] is not None
    ]
    scores_absents = [
        r["top1_reranker_score"] for r in report["details"]
        if not r.get("reference_invalide") and not r["in_pool"]
        and r["top1_reranker_score"] is not None
    ]

    def _mediane(valeurs: list[float]) -> float | None:
        if not valeurs:
            return None
        s = sorted(valeurs)
        n = len(s)
        milieu = n // 2
        return s[milieu] if n % 2 == 1 else (s[milieu - 1] + s[milieu]) / 2

    print(f"\n  [SEPARATION DES SCORES]")
    print(
        f"  Bonne fiche en rang 1 (in-pool)      : "
        f"n={len(scores_rang1)}, médiane={_mediane(scores_rang1)}, "
        f"moyenne={(_moyenne_liste(scores_rang1))}"
    )
    print(
        f"  top1_score si id_reference absente   : "
        f"n={len(scores_absents)}, médiane={_mediane(scores_absents)}, "
        f"moyenne={(_moyenne_liste(scores_absents))}"
    )
    print()


def _moyenne_liste(valeurs: list[float]) -> float | None:
    return sum(valeurs) / len(valeurs) if valeurs else None


def _fmt_score(valeur: float | None) -> str:
    return f"{valeur:.3f}" if valeur is not None else "-"


def print_breakdown_negatifs_par_type(
    negatifs_results: list[dict[str, Any]], seuil: float = RAG_CONFIDENCE_THRESHOLD
) -> None:
    """
    Score médian/max et taux de faux positifs au seuil de confiance, groupés
    par "type" (negatif_tickets_reels/negatif_synthetique/negatif_ood) — les
    negatif_ood (hors-scope non-facturation) sont le cas le plus proche d'un
    vrai usage en production, à surveiller séparément des négatifs synthétiques.
    """
    types = sorted({r["type"] for r in negatifs_results})

    print(f"\n=== BREAKDOWN NÉGATIFS PAR TYPE ===")
    print(f"  {'Type':<24}| {'N':>3} | {'Score médian':>12} | {'Score max':>9} | {'FP@' + f'{seuil:.2f}':>8}")
    for type_val in types:
        subset = [r for r in negatifs_results if r["type"] == type_val]
        scores = sorted(r["top1_reranker_score"] for r in subset if r["top1_reranker_score"] is not None)
        fp = sum(1 for r in subset if r["top1_reranker_score"] is not None and r["top1_reranker_score"] > seuil)
        median = _percentile(scores, 0.5) if scores else None
        score_max = scores[-1] if scores else None
        print(
            f"  {type_val:<24}| {len(subset):>3} | {_fmt_score(median):>12} | "
            f"{_fmt_score(score_max):>9} | {fp}/{len(subset)}"
        )


def print_negatifs_summary(
    stats: dict[str, Any], negatifs_results: list[dict[str, Any]]
) -> None:
    print(f"\n=== RAPPORT NÉGATIFS (gate d'abstention) ===")
    print(f"  Négatifs traités : {stats['nb_negatifs']} (scores valides : {stats['nb_scores_valides']})")

    print(f"\n  Distribution du top1_reranker_score :")
    print(
        f"    min={_fmt_score(stats['top1_score_min'])}  "
        f"p25={_fmt_score(stats['top1_score_p25'])}  "
        f"médiane={_fmt_score(stats['top1_score_median'])}  "
        f"p75={_fmt_score(stats['top1_score_p75'])}  "
        f"max={_fmt_score(stats['top1_score_max'])}  "
        f"moyenne={_fmt_score(stats['top1_score_moyenne'])}"
    )

    print(f"\n  Taux de faux positifs (négatif qui dépasserait le seuil) :")
    for seuil, taux in stats["taux_faux_positifs"].items():
        print(f"    seuil={seuil:.2f} : {taux:.1%}")

    dangereux = sorted(
        (r for r in negatifs_results if r["top1_reranker_score"] is not None),
        key=lambda r: r["top1_reranker_score"],
        reverse=True,
    )[:3]
    print(f"\n  Top 3 des négatifs les plus dangereux (top1_reranker_score le plus élevé) :")
    if not dangereux:
        print(f"    (aucun score valide)")
    else:
        for r in dangereux:
            print(
                f"    {r['id_test']} | score={_fmt_score(r['top1_reranker_score'])} "
                f"| {r['question_client']}"
            )
    print()

    print_breakdown_negatifs_par_type(negatifs_results)


def print_grid_search_summary(resultats_grille: list[dict[str, Any]]) -> None:
    print(f"\n[GRID SEARCH — poids RRF w_dense / w_sparse = 1 - w_dense]")
    print(
        f"  {'w_dense':>8} {'w_sparse':>9} {'R@1':>7} {'R@3':>7} {'R@5':>7} "
        f"{'R@10':>7} {'P@1':>7} {'MRR':>8}"
    )
    for r in resultats_grille:
        print(
            f"  {r['w_dense']:>8.1f} {r['w_sparse']:>9.1f} "
            f"{r['recall_at_1_moyen']:>6.1%} {r['recall_at_3_moyen']:>6.1%} "
            f"{r['recall_at_5_moyen']:>6.1%} {r['recall_at_10_moyen']:>6.1%} "
            f"{r['precision_at_1_moyen']:>6.1%} {r['mrr']:>8.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Évaluation retrieval hybride (recall@1/3/5/10, precision@1, MRR) sur golden dataset"
    )
    parser.add_argument(
        "--dataset-path", "--golden", dest="dataset_path", type=str, default=None,
        help=(
            "Chemin du golden dataset à utiliser. Par défaut : sélection automatique "
            "du fichier tests/golden_dataset*.json de version la plus élevée."
        ),
    )
    parser.add_argument("--ingestion-config", type=str, default=DEFAULT_INGESTION_CONFIG_PATH)
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Recherche en grille du poids RRF (w_dense/w_sparse) au lieu d'une évaluation simple",
    )
    parser.add_argument(
        "--grid-w-dense", type=str, default=None,
        help="Liste de poids w_dense séparés par des virgules (défaut : 0.5,0.6,0.7,0.8,0.9)",
    )
    parser.add_argument("--grid-report", type=str, default=DEFAULT_GRID_REPORT_PATH)
    parser.add_argument(
        "--dense-only", action="store_true",
        help="Évaluation dense-only (sans fusion RRF ni BM25) au lieu du retrieval hybride",
    )
    parser.add_argument("--dense-only-report", type=str, default=DEFAULT_DENSE_ONLY_REPORT_PATH)
    parser.add_argument(
        "--rerank", action="store_true",
        help="Évaluation avec reranking cross-encoder (2e étage, BAAI/bge-reranker-v2-m3) sur le pool hybride",
    )
    parser.add_argument("--first-stage-k", type=int, default=DEFAULT_FIRST_STAGE_K)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--rerank-report", type=str, default=DEFAULT_RERANK_REPORT_CSV_PATH)
    parser.add_argument(
        "--dump-top1-positifs", action="store_true",
        help="Exporte les résultats rang-1 pour questions positives dans tests/top1_positifs_dump.csv",
    )
    args = parser.parse_args()

    try:
        golden_path = resolve_golden_dataset(args.dataset_path)
        log_golden_dataset_selection(golden_path)

        if args.rerank:
            run_rerank(
                golden_path, args.ingestion_config,
                first_stage_k=args.first_stage_k, top_n=args.top_n,
                report_csv_path=args.rerank_report,
                dump_top1_positifs=args.dump_top1_positifs,
            )
        elif args.grid_search:
            poids = (
                [float(w) for w in args.grid_w_dense.split(",")]
                if args.grid_w_dense else None
            )
            resultats_grille = grid_search_rrf_weights(golden_path, args.ingestion_config, poids)
            print_grid_search_summary(resultats_grille)
            Path(args.grid_report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.grid_report, "w", encoding="utf-8") as f:
                json.dump(resultats_grille, f, ensure_ascii=False, indent=2)
        elif args.dense_only:
            dense_report = run_dense_only(golden_path, args.ingestion_config, args.dense_only_report)

            hybride_report = None
            if Path(args.report).exists():
                with open(args.report, "r", encoding="utf-8") as f:
                    hybride_report = json.load(f)
            print_comparaison_hybride_vs_dense(hybride_report, dense_report)

            inspect_groupe_b_dense_only(golden_path, args.ingestion_config)
        else:
            run(golden_path, args.ingestion_config, args.report)
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
