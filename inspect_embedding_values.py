"""
Script d'inspection : calcule et affiche les valeurs réelles des vecteurs
d'embedding (dense, requête + candidats) et les scores de similarité cosinus,
pour vérifier "à la main" ce que fait Retriever.retrieve() en interne.

Réutilise Embedder (ai_engine/rag/embedder.py) pour l'encodage de la requête —
même modèle, mêmes paramètres (normalize_embeddings=True) que la production —
et récupère les vecteurs dense réellement stockés dans Qdrant
(with_vectors=True) pour recalculer le cosinus manuellement (numpy), en le
comparant au score renvoyé par Qdrant (doit correspondre : distance=Cosine).

Usage :
    python inspect_embedding_values.py "ma question" --top 5
    python inspect_embedding_values.py "ma question" --report tests/embedding_values_report.json

Écrit un rapport JSON complet (vecteurs 768d intégraux, pas tronqués) et
affiche un résumé lisible à l'écran (vecteurs tronqués aux 5 premières/
dernières valeurs, cosinus en clair).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from qdrant_client import QdrantClient

from ai_engine.config import get_settings
from ai_engine.rag.embedder import Embedder
from data_pipeline.ingestion.ingest_qr_technique import DENSE_VECTOR_NAME

COLLECTION_NAME = "connaissances_modules"
DEFAULT_TOP = 5
DEFAULT_REPORT_PATH = "tests/embedding_values_report.json"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Affiche/écrit les valeurs des vecteurs d'embedding et des scores cosinus"
    )
    parser.add_argument("query", type=str, help="Requête à encoder")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="Nombre de candidats dense inspectés")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH, help="Chemin du rapport JSON complet")
    args = parser.parse_args()

    print("=" * 80)
    print("VALEURS DES EMBEDDINGS ET SCORES COSINUS")
    print("=" * 80)
    print(f"\nRequête : {args.query!r}")

    embedder = Embedder()
    query_vector = embedder.encode(args.query)

    norme_query = float(np.linalg.norm(query_vector))
    print(f"\nVecteur requête : dimension={len(query_vector)}, norme L2={norme_query:.6f}")
    print(f"  5 premières valeurs : {[round(v, 6) for v in query_vector[:5]]}")
    print(f"  5 dernières valeurs : {[round(v, 6) for v in query_vector[-5:]]}")

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    print(f"\nRecherche dense (Qdrant, collection={COLLECTION_NAME}, top={args.top})…")
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=(DENSE_VECTOR_NAME, query_vector),
        limit=args.top,
        with_payload=True,
        with_vectors=True,
    )

    candidats_rapport = []
    print(f"\n{'rang':<5} {'score_qdrant':>13} {'cosinus_recalcule':>18} {'ecart':>10}  categorie / question")
    for rang, hit in enumerate(hits, start=1):
        candidat_vector = hit.vector[DENSE_VECTOR_NAME] if isinstance(hit.vector, dict) else hit.vector
        cos_recalcule = cosine_similarity(query_vector, candidat_vector)
        ecart = abs(hit.score - cos_recalcule)
        categorie = hit.payload.get("categorie", "?")
        question = (hit.payload.get("question") or "")[:60]

        print(
            f"{rang:<5} {hit.score:>13.6f} {cos_recalcule:>18.6f} {ecart:>10.2e}  "
            f"{categorie} / {question}"
        )

        candidats_rapport.append({
            "rang": rang,
            "id": str(hit.id),
            "categorie": categorie,
            "question": hit.payload.get("question"),
            "score_qdrant": hit.score,
            "cosinus_recalcule": cos_recalcule,
            "ecart_score_vs_recalcule": ecart,
            "vecteur_dense_candidat": list(candidat_vector),
        })

    rapport = {
        "query": args.query,
        "modele_embedding": embedder.model_name,
        "vecteur_requete": {
            "dimension": len(query_vector),
            "norme_l2": norme_query,
            "valeurs": list(query_vector),
        },
        "candidats": candidats_rapport,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Rapport complet (vecteurs 768d intégraux) écrit dans : {args.report}")


if __name__ == "__main__":
    main()
