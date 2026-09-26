"""
test_motscles_evaluate.py
Évalue le golden dataset contre la collection Qdrant DE TEST
"connaissances_modules_test_motscles" (formule texte_pour_embedding avec
mots_cles), et compare à la baseline production (tests/retrieval_eval_report.json).

Pourquoi ce script est séparé plutôt qu'un simple --collection sur
evaluate_retrieval.py : evaluate_retrieval.py::build_reference_corpus()
reconstruit le vocabulaire BM25 en appelant prepare_entries() de
ingest_qr_technique.py, qui est CODÉ EN DUR sur la formule ORIGINALE
(sans mots_cles). Un simple changement de collection_name ne suffirait pas :
le vocabulaire BM25 (indices/IDF) doit être reconstruit avec la MÊME formule
que celle utilisée pour ingérer les vecteurs sparse de la collection de test
(cf. test_motscles_ingest_variant.py::build_texte_embedding_avec_mots_cles),
sinon les indices du vecteur sparse requête ne correspondraient pas à ceux
stockés. Modifier evaluate_retrieval.py pour prendre en charge une formule
paramétrable aurait touché du code de production pour un test ponctuel — un
script séparé isole ce risque sans y toucher.

Réutilise (sans dupliquer) les primitives génériques de
ai_engine/rag/evaluate_retrieval.py : dense_only_search, hybrid_search,
reciprocal_rank_fusion (implicite dans hybrid_search).

Usage :
    python test_motscles_evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ai_engine.config import get_settings
from ai_engine.rag.evaluate_retrieval import (
    DEFAULT_GOLDEN_PATH,
    RECALL_DEPTHS,
    TOP_K_MAX,
    dense_only_search,
    hybrid_search,
    load_golden_dataset,
)
from data_pipeline.ingestion.ingest_qr_technique import (
    build_bm25_index,
    build_point_id,
    load_config,
    load_source,
    tokenize,
    validate_entry,
    vectorize_bm25,
)
from test_motscles_ingest_variant import build_texte_embedding_avec_mots_cles

INGESTION_CONFIG_PATH = "data_pipeline/config/ingestion_config_test_motscles.json"
BASELINE_REPORT_PATH = "tests/retrieval_eval_report.json"
OUTPUT_REPORT_PATH = "tests/retrieval_eval_motscles_variant_report.json"


# ── Reconstruction du corpus de référence AVEC LA FORMULE VARIANTE ──────────

def build_reference_corpus_avec_mots_cles(ingestion_config_path: str) -> tuple[list[str], dict[str, Any]]:
    """
    Reconstruit le corpus texte EXACTEMENT comme lors de l'ingestion de test
    (test_motscles_ingest_variant.py) : même formule avec mots_cles, même
    troncature. Nécessaire pour que le vocabulaire BM25 (indices/IDF) de la
    requête corresponde à celui utilisé pour les vecteurs sparse déjà stockés
    dans connaissances_modules_test_motscles.
    """
    config = load_config(ingestion_config_path)
    raw_entries = load_source(config["chemin_source"])

    texts = []
    for entry in raw_entries:
        try:
            validate_entry(entry)
            texts.append(build_texte_embedding_avec_mots_cles(entry, config["longueur_troncature_reponse"]))
        except Exception:
            continue  # cohérent avec prepare_entries() : entrée invalide exclue, journalisée à l'ingestion

    return texts, config


# ── Évaluation par question (miroir de evaluate_question, collection paramétrable) ──

def evaluate_question_motscles(
    question: dict[str, Any],
    model: SentenceTransformer,
    bm25,
    vocab: dict[str, int],
    client: QdrantClient,
    collection_name: str,
) -> dict[str, Any]:
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

    top_results = hybrid_search(client, collection_name, dense_query, sparse_query, limit=TOP_K_MAX)
    top_ids = [point_id for point_id, _ in top_results]

    trouve = ground_truth_id in top_ids
    rang_trouve = top_ids.index(ground_truth_id) + 1 if trouve else None
    precision_1 = 1 if top_ids and top_ids[0] == ground_truth_id else 0
    reciprocal_rank = 1.0 / rang_trouve if rang_trouve is not None else 0.0

    recalls = {
        f"recall_at_{profondeur}": 1 if (rang_trouve is not None and rang_trouve <= profondeur) else 0
        for profondeur in RECALL_DEPTHS
    }

    return {
        "id_test": question["id_test"],
        "categorie": question["categorie"],
        "reference_invalide": False,
        "rang_trouve": rang_trouve,
        **recalls,
        "precision_at_1": precision_1,
        "reciprocal_rank": reciprocal_rank,
    }


def _moyenne(scores: list[dict[str, Any]], cle: str) -> float:
    return sum(r[cle] for r in scores) / len(scores) if scores else 0.0


# ── Pipeline principal ────────────────────────────────────────────────────────

def run() -> dict[str, Any]:
    questions = load_golden_dataset(DEFAULT_GOLDEN_PATH)
    print(f"[INFO] {len(questions)} questions chargées depuis {DEFAULT_GOLDEN_PATH}")

    print("[INFO] Reconstruction du corpus de référence (formule AVEC mots_cles)…")
    texts, ingestion_config = build_reference_corpus_avec_mots_cles(INGESTION_CONFIG_PATH)
    bm25, vocab = build_bm25_index(texts)

    print(f"[INFO] Chargement du modèle : {ingestion_config['modele_embedding']}")
    model = SentenceTransformer(ingestion_config["modele_embedding"])

    settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {settings.qdrant_url}")
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    collection_name = ingestion_config["collection_name"]
    print(f"[INFO] Collection évaluée : {collection_name}")

    results = [
        evaluate_question_motscles(q, model, bm25, vocab, client, collection_name)
        for q in questions
    ]

    scores = [r for r in results if not r["reference_invalide"]]
    exclus = [r for r in results if r["reference_invalide"]]

    report = {
        "collection_name": collection_name,
        "formule": "categorie. mots_cles. question. question. reponse[:400]",
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

    Path(OUTPUT_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def load_baseline() -> dict[str, Any] | None:
    if not Path(BASELINE_REPORT_PATH).exists():
        return None
    with open(BASELINE_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison(baseline: dict[str, Any] | None, test: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("COMPARAISON AGRÉGÉE — baseline (connaissances_modules) vs test (mots_cles inclus)")
    print("=" * 80)

    if baseline is None:
        print("  [ATTENTION] Rapport baseline introuvable — comparaison ignorée.")
        return

    metriques = [
        ("Recall@1", "recall_at_1_moyen"),
        ("Recall@3", "recall_at_3_moyen"),
        ("Recall@5", "recall_at_5_moyen"),
        ("Recall@10", "recall_at_10_moyen"),
        ("Precision@1", "precision_at_1_moyen"),
        ("MRR", "mrr"),
    ]
    print(f"  {'Métrique':<12} {'Baseline':>10} {'Test (motscles)':>17} {'Écart':>10}")
    for label, cle in metriques:
        v_base = baseline.get(cle, 0.0)
        v_test = test.get(cle, 0.0)
        ecart = v_test - v_base
        if cle == "mrr":
            print(f"  {label:<12} {v_base:>10.4f} {v_test:>17.4f} {ecart:>+10.4f}")
        else:
            print(f"  {label:<12} {v_base:>9.1%} {v_test:>16.1%} {ecart:>+9.1%}")

    print("\n" + "=" * 80)
    print("DÉTAIL PAR QUESTION — baseline vs test")
    print("=" * 80)
    baseline_par_id = {d["id_test"]: d for d in baseline.get("details", []) if not d.get("reference_invalide")}
    test_par_id = {d["id_test"]: d for d in test.get("details", []) if not d.get("reference_invalide")}

    print(f"  {'id_test':<22} {'categorie':<20} {'rang_base':>10} {'rang_test':>10} {'evolution':>12}")
    regressions = []
    ameliorations = []
    for id_test in sorted(set(baseline_par_id) | set(test_par_id)):
        b = baseline_par_id.get(id_test)
        t = test_par_id.get(id_test)
        categorie = (t or b)["categorie"]
        rang_b = b["rang_trouve"] if b else None
        rang_t = t["rang_trouve"] if t else None

        rang_b_str = str(rang_b) if rang_b is not None else "-"
        rang_t_str = str(rang_t) if rang_t is not None else "-"

        if rang_b is None and rang_t is None:
            evolution = "toujours absent"
        elif rang_b is None and rang_t is not None:
            evolution = "GAIN (nouv. trouvé)"
            ameliorations.append(id_test)
        elif rang_b is not None and rang_t is None:
            evolution = "REGRESSION (perdu)"
            regressions.append(id_test)
        elif rang_t < rang_b:
            evolution = f"amélioré (+{rang_b - rang_t})"
            ameliorations.append(id_test)
        elif rang_t > rang_b:
            evolution = f"dégradé (-{rang_t - rang_b})"
            regressions.append(id_test)
        else:
            evolution = "inchangé"

        print(f"  {id_test:<22} {categorie:<20} {rang_b_str:>10} {rang_t_str:>10} {evolution:>12}")

    print(f"\n  Questions améliorées ou nouvellement trouvées : {len(ameliorations)} -> {ameliorations}")
    print(f"  Questions dégradées ou perdues (régression)    : {len(regressions)} -> {regressions}")


def main():
    print("=" * 80)
    print("ÉVALUATION VARIANTE MOTS_CLES — connaissances_modules_test_motscles")
    print("=" * 80)

    test_report = run()
    baseline_report = load_baseline()
    print_comparison(baseline_report, test_report)

    print(f"\n[INFO] Rapport de test sauvegardé dans {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
