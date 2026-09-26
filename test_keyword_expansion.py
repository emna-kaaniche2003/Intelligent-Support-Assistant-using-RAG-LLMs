"""
test_keyword_expansion.py
Script de test ISOLE (lecture seule) — mesure le plafond de gain possible
d'une expansion de requête par mots_cles, pour les questions du golden dataset
absentes du top-20 en dense-only ET en hybride.

Ne modifie ni les points Qdrant, ni evaluate_retrieval.py, ni aucun fichier
de production. Réutilise (n'importe, ne duplique pas) les fonctions déjà
validées de ai_engine/rag/evaluate_retrieval.py et
data_pipeline/ingestion/ingest_qr_technique.py.

Usage :
    python test_keyword_expansion.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ai_engine.config import get_settings
from ai_engine.rag.evaluate_retrieval import (
    DEFAULT_GOLDEN_PATH,
    DEFAULT_INGESTION_CONFIG_PATH,
    RECALL_DEPTHS,
    build_reference_corpus,
    dense_only_search,
    hybrid_search,
    load_golden_dataset,
    load_ingestion_config,
)
from data_pipeline.ingestion.ingest_qr_technique import (
    build_bm25_index,
    build_point_id,
    tokenize,
    vectorize_bm25,
)

INSPECTION_LIMIT = 20  # profondeur de recherche pour cette inspection ciblée


# ── Étape 0 : identifier les catégories absentes du top-20 en dense ET hybride ──

def find_double_failures(
    golden_path: str,
    ingestion_config_path: str,
    limit: int = INSPECTION_LIMIT,
) -> list[dict[str, Any]]:
    """
    Balaie les 20 questions du golden dataset et identifie celles dont le
    ground truth est absent du top-`limit` à la fois en dense-only et en
    hybride (RRF). Confirme (ou infirme) l'existence d'une 3e catégorie au-delà
    de Devis/Forfaits déjà repérées.
    """
    questions = load_golden_dataset(golden_path)
    ingestion_config = load_ingestion_config(ingestion_config_path)
    texts, _ = build_reference_corpus(ingestion_config_path)
    bm25, vocab = build_bm25_index(texts)

    model = SentenceTransformer(ingestion_config["modele_embedding"])
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    collection_name = ingestion_config["collection_name"]

    double_failures = []
    for question in questions:
        ground_truth_id = build_point_id(question["id_reference"], question["source_type"])
        existants = client.retrieve(collection_name=collection_name, ids=[ground_truth_id])
        if not existants:
            continue

        query_text = question["question_client"]
        dense_query = model.encode(
            query_text, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        sparse_query = vectorize_bm25(tokenize(query_text), bm25, vocab)

        dense_top = [pid for pid, _ in dense_only_search(client, collection_name, dense_query, limit=limit)]
        hybride_top = [
            pid for pid, _ in hybrid_search(
                client, collection_name, dense_query, sparse_query, limit=limit
            )
        ]

        absent_dense = ground_truth_id not in dense_top
        absent_hybride = ground_truth_id not in hybride_top

        if absent_dense and absent_hybride:
            double_failures.append({
                "id_test": question["id_test"],
                "categorie": question["categorie"],
                "ground_truth_id": ground_truth_id,
                "question": question,
            })

    return double_failures


# ── Étape 1-3 : requête augmentée par mots_cles ──────────────────────────────

def get_target_point_payload(
    client: QdrantClient, collection_name: str, ground_truth_id: str
) -> dict[str, Any]:
    points = client.retrieve(collection_name=collection_name, ids=[ground_truth_id], with_payload=True)
    if not points:
        raise ValueError(f"Point {ground_truth_id} introuvable dans Qdrant")
    return points[0].payload


def build_augmented_query(question_client: str, mots_cles: list[str]) -> str:
    """<question_client> <mot_cle_1> <mot_cle_2> ..."""
    return f"{question_client} {' '.join(mots_cles)}"


def rank_of(ground_truth_id: str, top_ids: list[str]) -> int | None:
    return top_ids.index(ground_truth_id) + 1 if ground_truth_id in top_ids else None


def evaluate_query_variant(
    label: str,
    query_text: str,
    ground_truth_id: str,
    model: SentenceTransformer,
    bm25,
    vocab: dict[str, int],
    client: QdrantClient,
    collection_name: str,
    limit: int = INSPECTION_LIMIT,
) -> dict[str, Any]:
    dense_query = model.encode(
        query_text, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    sparse_query = vectorize_bm25(tokenize(query_text), bm25, vocab)

    dense_top_ids = [pid for pid, _ in dense_only_search(client, collection_name, dense_query, limit=limit)]
    hybride_top_ids = [
        pid for pid, _ in hybrid_search(client, collection_name, dense_query, sparse_query, limit=limit)
    ]

    return {
        "label": label,
        "query_text": query_text,
        "rang_dense": rank_of(ground_truth_id, dense_top_ids),
        "rang_hybride": rank_of(ground_truth_id, hybride_top_ids),
    }


def format_rang(rang: int | None, limit: int) -> str:
    return f"rang {rang}" if rang is not None else f"absent du top-{limit}"


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("TEST D'EXPANSION DE REQUÊTE PAR MOTS_CLES (plafond de gain, script isolé)")
    print("=" * 80)

    print("\n[ÉTAPE 0] Balayage des 20 questions : absentes du top-20 en dense ET en hybride ?")
    double_failures = find_double_failures(DEFAULT_GOLDEN_PATH, DEFAULT_INGESTION_CONFIG_PATH, limit=INSPECTION_LIMIT)

    if not double_failures:
        print("  Aucune question n'est absente du top-20 dans les deux modes.")
    else:
        print(f"  {len(double_failures)} question(s) absente(s) du top-{INSPECTION_LIMIT} en dense ET en hybride :")
        for d in double_failures:
            print(f"    - {d['id_test']} | categorie={d['categorie']}")

    id_tests_cibles = {"gd_facturation_006", "gd_facturation_011"}
    id_tests_trouves = {d["id_test"] for d in double_failures}
    if id_tests_trouves != id_tests_cibles:
        manquants = id_tests_cibles - id_tests_trouves
        supplementaires = id_tests_trouves - id_tests_cibles
        if manquants:
            print(f"  [ATTENTION] Attendu(s) mais non confirmé(s) en échec double : {manquants}")
        if supplementaires:
            print(f"  [NOUVEAU] 3e catégorie (ou plus) confirmée en échec double : {supplementaires}")

    # Setup partagé pour l'étape mots_cles (réutilise le même modèle/bm25/client)
    ingestion_config = load_ingestion_config(DEFAULT_INGESTION_CONFIG_PATH)
    texts, _ = build_reference_corpus(DEFAULT_INGESTION_CONFIG_PATH)
    bm25, vocab = build_bm25_index(texts)
    model = SentenceTransformer(ingestion_config["modele_embedding"])
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    collection_name = ingestion_config["collection_name"]

    print("\n" + "=" * 80)
    print("[ÉTAPES 1-3] Requête augmentée par mots_cles — gd_facturation_006 et gd_facturation_011")
    print("=" * 80)

    resultats = []
    for d in double_failures:
        if d["id_test"] not in id_tests_cibles:
            continue

        question = d["question"]
        ground_truth_id = d["ground_truth_id"]
        payload = get_target_point_payload(client, collection_name, ground_truth_id)
        mots_cles = payload.get("mots_cles") or []

        print(f"\n--- {d['id_test']} | categorie={d['categorie']} ---")
        print(f"  Question golden dataset : {question['question_client']}")
        print(f"  Question Qdrant (verite terrain) : {payload.get('question', '')[:120]}")
        print(f"  mots_cles du point cible : {mots_cles}")

        if not mots_cles:
            print("  [ATTENTION] Aucun mot_cle sur ce point — expansion impossible, test ignoré.")
            continue

        query_originale = question["question_client"]
        query_augmentee = build_augmented_query(query_originale, mots_cles)
        print(f"  Requête augmentée : {query_augmentee}")

        avant = evaluate_query_variant(
            "originale", query_originale, ground_truth_id, model, bm25, vocab, client, collection_name
        )
        apres = evaluate_query_variant(
            "augmentee", query_augmentee, ground_truth_id, model, bm25, vocab, client, collection_name
        )

        print(f"\n  Résultat AVANT expansion :")
        print(f"    dense-only : {format_rang(avant['rang_dense'], INSPECTION_LIMIT)}")
        print(f"    hybride    : {format_rang(avant['rang_hybride'], INSPECTION_LIMIT)}")
        print(f"  Résultat APRES expansion (mots_cles concaténés) :")
        print(f"    dense-only : {format_rang(apres['rang_dense'], INSPECTION_LIMIT)}")
        print(f"    hybride    : {format_rang(apres['rang_hybride'], INSPECTION_LIMIT)}")

        resultats.append({
            "id_test": d["id_test"],
            "categorie": d["categorie"],
            "mots_cles": mots_cles,
            "avant": avant,
            "apres": apres,
        })

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    for r in resultats:
        rang_apres_hybride = r["apres"]["rang_hybride"]
        rang_apres_dense = r["apres"]["rang_dense"]
        meilleur_rang = min(
            [x for x in (rang_apres_hybride, rang_apres_dense) if x is not None],
            default=None,
        )
        if meilleur_rang is not None and meilleur_rang <= 3:
            verdict = "GAIN NET (top-3 atteint avec les mots_cles exacts) -> expansion de requete = piste rentable"
        elif meilleur_rang is not None:
            verdict = f"GAIN PARTIEL (rang {meilleur_rang}, hors top-3) -> expansion utile mais insuffisante seule"
        else:
            verdict = "AUCUN GAIN (toujours absent du top-20 meme avec les mots_cles exacts) -> signal grave, embedding lui-meme ne rapproche pas -> piste bge-m3/reranker a privilegier"
        print(f"  {r['id_test']} ({r['categorie']}) : {verdict}")
    print()


if __name__ == "__main__":
    main()
