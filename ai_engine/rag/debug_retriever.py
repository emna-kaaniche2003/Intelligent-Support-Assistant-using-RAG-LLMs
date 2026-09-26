"""
ai_engine/rag/debug_retriever.py
Instrumentation de diagnostic pour Retriever.retrieve() — À LANCER MANUELLEMENT,
ne fait pas partie de l'interface de production (ai_engine/rag/retriever.py).

Affiche, dans l'ordre d'exécution du pipeline retrieve() : vecteur dense de la
requête, vecteur sparse BM25, candidats dense/sparse avant fusion RRF, pool
après fusion, résultats après reranking, puis la décision du gate d'abstention.

Origine : investigation S5 d'un écart entre evaluate_retrieval.py --rerank et
Retriever.retrieve() sur gd_facturation_001 (id_reference=286) — cf. CLAUDE.md
§ Évaluation RAG, Étape 6. Conclusion de cette investigation : comportement
correct de retriever.py/embedder.py, pas un bug (sensibilité normale à la
reformulation de requête, cf. golden_dataset_v4.json / gd_facturation_061).

Usage :
    python ai_engine/rag/debug_retriever.py "ma question" \
        --module Facturation --priorite Assistance \
        --id-reference 286 --source-type qr_technique
"""

from __future__ import annotations

import argparse
import math
from typing import Any

from ai_engine.rag.evaluate_retrieval import fetch_dense_sparse_candidates, reciprocal_rank_fusion
from ai_engine.rag.retriever import Retriever
from data_pipeline.ingestion.ingest_qr_technique import build_point_id, tokenize, vectorize_bm25


def _print_dense_vector(dense_query: list[float]) -> None:
    norme = math.sqrt(sum(v * v for v in dense_query))
    premieres = ", ".join(f"{v:.3f}" for v in dense_query[:5])
    dernieres = ", ".join(f"{v:.3f}" for v in dense_query[-5:])
    print("\n─── ÉTAPE 0 : VECTEUR DENSE DE LA REQUÊTE ───")
    print(f"Norme L2 du vecteur : {norme:.6f}  (doit être ≈ 1.0 si normalize=True)")
    print(f"Dimension : {len(dense_query)}")
    print(f"5 premières valeurs : [{premieres}]")
    print(f"5 dernières valeurs : [{dernieres}]")


def _print_sparse_vector(sparse_query, vocab: dict[str, int]) -> None:
    indices = list(sparse_query.indices)
    values = list(sparse_query.values)
    print("\n─── ÉTAPE 1 : VECTEUR SPARSE BM25 DE LA REQUÊTE ───")
    print(f"Nombre d'indices non-nuls : {len(indices)}")
    if not indices:
        print("ANOMALIE : vocabulaire BM25 vide ou incompatible (0 indices non-nuls)")
        return
    idx_vers_token = {idx: token for token, idx in vocab.items()}
    paires = sorted(zip(indices, values), key=lambda iv: iv[1], reverse=True)[:10]
    print("Top 10 tokens (indice → valeur IDF) :")
    for idx, val in paires:
        token = idx_vers_token.get(idx, "?")
        print(f"  indice_{idx} ({token}) → {val:.3f}")


def _fetch_payloads(client, collection_name: str, ids: list[str]) -> dict[str, Any]:
    if not ids:
        return {}
    points = client.retrieve(collection_name=collection_name, ids=ids, with_payload=True)
    return {str(p.id): p.payload for p in points}


def _print_scored_points(titre: str, hits, payload_par_id: dict[str, Any], score_label: str) -> None:
    print(f"\n─── {titre} ───")
    if not hits:
        print("(liste vide → ANOMALIE)")
        return
    print(f"Top {min(5, len(hits))} résultats Qdrant :")
    for rang, hit in enumerate(hits[:5], start=1):
        point_id = str(hit.id)
        payload = payload_par_id.get(point_id, {})
        categorie = payload.get("categorie", "?")
        question = (payload.get("question") or "")[:60]
        print(
            f"  rang {rang} : id={point_id} | {score_label}={hit.score:.4f} "
            f"| categorie={categorie} | question[:60]={question}"
        )


def _print_pool(
    fused: list[tuple[str, float]],
    limit: int,
    candidates: list[dict[str, Any]],
    ground_truth_id: str | None,
) -> None:
    pool = fused[:limit]
    payload_par_id = {c["id"]: c["payload"] for c in candidates}
    print("\n─── ÉTAPE 4 : POOL APRÈS FUSION RRF (avant rerank) ───")
    print(f"Taille du pool : {len(pool)} candidats (attendu : RAG_FIRST_STAGE_K={limit})")
    print(f"Top {min(5, len(pool))} après RRF :")
    for rang, (point_id, score) in enumerate(pool[:5], start=1):
        payload = payload_par_id.get(point_id, {})
        categorie = payload.get("categorie", "?")
        question = (payload.get("question") or "")[:60]
        print(
            f"  rang {rang} : id={point_id} | score_rrf={score:.4f} "
            f"| categorie={categorie} | question[:60]={question}"
        )
    if ground_truth_id is None:
        return
    pool_ids = [pid for pid, _ in pool]
    if ground_truth_id in pool_ids:
        print(f"ground_truth présent dans le pool ? OUI (rang {pool_ids.index(ground_truth_id) + 1})")
    else:
        print("ground_truth présent dans le pool ? NON")


def _print_rerank(candidats_rerankes: list[dict[str, Any]], ground_truth_id: str | None) -> None:
    print("\n─── ÉTAPE 5 : RÉSULTATS APRÈS RERANK ───")
    print(f"Top {min(3, len(candidats_rerankes))} après reranking :")
    ids_rerankes = [c["id"] for c in candidats_rerankes]
    for rang, c in enumerate(candidats_rerankes[:3], start=1):
        payload = c.get("payload", {})
        categorie = payload.get("categorie", "?")
        question = (payload.get("question") or "")[:60]
        print(
            f"  rang {rang} : id={c['id']} | reranker_score={c['reranker_score']:.4f} "
            f"| categorie={categorie} | question[:60]={question}"
        )
    if ground_truth_id is None:
        return
    if ground_truth_id in ids_rerankes:
        rang = ids_rerankes.index(ground_truth_id) + 1
        score = candidats_rerankes[rang - 1]["reranker_score"]
        print(f"ground_truth présent ? OUI (rang {rang}, score {score:.4f})")
    else:
        print("ground_truth présent ? NON")


def debug_retrieve(
    retriever: Retriever,
    query: str,
    module: str | None = None,
    priorite: str | None = None,
    top_n: int = 3,
    id_reference: Any = None,
    source_type: str | None = None,
):
    """
    Rejoue le pipeline de Retriever.retrieve() étape par étape, avec affichage
    de diagnostic à chaque étape. N'appelle PAS retrieve() (interface de
    production sans instrumentation) : reproduit les mêmes appels sur les
    composants déjà chargés du retriever (embedder, reranker, bm25, client).

    id_reference/source_type (optionnels) : si fournis, localise la
    vérité-terrain correspondante dans le pool/rerank (build_point_id).
    """
    settings = retriever.settings
    ground_truth_id = (
        build_point_id(id_reference, source_type) if id_reference is not None else None
    )

    if settings.rag_module_filter_enabled:
        if module is not None and module not in settings.rag_scope_modules:
            print(f"\nPré-qualification : module_hors_scope ({module}) → escalade avant retrieval.")
            return
        if priorite is not None and priorite not in settings.rag_scope_priorities:
            print(f"\nPré-qualification : priorite_hors_scope ({priorite}) → escalade avant retrieval.")
            return

    dense_query = retriever.embedder.encode(query)
    _print_dense_vector(dense_query)

    sparse_query = vectorize_bm25(tokenize(query), retriever.bm25, retriever.vocab)
    _print_sparse_vector(sparse_query, retriever.vocab)

    dense_results, sparse_results = fetch_dense_sparse_candidates(
        retriever.client,
        retriever.collection_name,
        dense_query,
        sparse_query,
        limit=settings.rag_first_stage_k,
    )
    debug_ids = list({str(h.id) for h in dense_results[:5]} | {str(h.id) for h in sparse_results[:5]})
    debug_payloads = _fetch_payloads(retriever.client, retriever.collection_name, debug_ids)
    _print_scored_points(
        "ÉTAPE 2 : CANDIDATS DENSE (avant RRF)", dense_results, debug_payloads, "score_cosinus"
    )
    _print_scored_points(
        "ÉTAPE 3 : CANDIDATS SPARSE BM25 (avant RRF)", sparse_results, debug_payloads, "score_bm25"
    )

    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    pool_ids = [point_id for point_id, _ in fused[: settings.rag_first_stage_k]]

    candidates: list[dict[str, Any]] = []
    if pool_ids:
        points = retriever.client.retrieve(
            collection_name=retriever.collection_name, ids=pool_ids, with_payload=True
        )
        payload_par_id = {str(p.id): p.payload for p in points}
        candidates = [
            {"id": point_id, "payload": payload_par_id[point_id]}
            for point_id in pool_ids
            if point_id in payload_par_id
        ]
    _print_pool(fused, settings.rag_first_stage_k, candidates, ground_truth_id)

    candidats_rerankes = retriever.reranker.rerank(query, candidates, top_n=top_n)
    _print_rerank(candidats_rerankes, ground_truth_id)

    top1_score = candidats_rerankes[0]["reranker_score"] if candidats_rerankes else 0.0
    answer_found = top1_score >= settings.rag_confidence_threshold
    print("\n─── DÉCISION GATE ───")
    print(
        f"top1_score={top1_score:.4f} | seuil={settings.rag_confidence_threshold:.2f} "
        f"| answer_found={answer_found}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic étape par étape du pipeline Retriever.retrieve()"
    )
    parser.add_argument("query", type=str, help="Question à tracer")
    parser.add_argument("--module", type=str, default=None)
    parser.add_argument("--priorite", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--id-reference", type=str, default=None,
        help="id_reference de la vérité-terrain à localiser dans le pool/rerank (optionnel)",
    )
    parser.add_argument(
        "--source-type", type=str, default="qr_technique",
        help="source_type associé à --id-reference (défaut : qr_technique)",
    )
    args = parser.parse_args()

    retriever = Retriever()
    debug_retrieve(
        retriever,
        args.query,
        module=args.module,
        priorite=args.priorite,
        top_n=args.top_n,
        id_reference=args.id_reference,
        source_type=args.source_type,
    )


if __name__ == "__main__":
    main()
