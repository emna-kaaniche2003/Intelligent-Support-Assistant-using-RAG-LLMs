"""
ai_engine/rag/retriever.py
Composant de production du retrieval RAG (Agent 4) : retrieval hybride
(dense + BM25 sparse, RRF 0.5/0.5) + reranking cross-encoder + gate
d'abstention — recette figée et validée sur golden_dataset_v3.json
(cf. CLAUDE.md § Évaluation RAG, Étapes 1-5).

Réutilise sans réécriture la logique de premier étage déjà validée dans
ai_engine/rag/evaluate_retrieval.py (fetch_dense_sparse_candidates,
reciprocal_rank_fusion, build_reference_corpus) — cf. TODO documenté dans
CLAUDE.md : refactor direct de ces fonctions, pas de réécriture "propre"
repensée. evaluate_retrieval.py reste la référence de comportement.

Instrumentation de diagnostic (vecteurs, pool avant/après RRF, rerank) :
voir ai_engine/rag/debug_retriever.py, à lancer manuellement — pas dans
l'interface de production retrieve().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient

from ai_engine.config import get_settings
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.evaluate_retrieval import (
    DEFAULT_INGESTION_CONFIG_PATH,
    build_reference_corpus,
    fetch_dense_sparse_candidates,
    reciprocal_rank_fusion,
)
from ai_engine.rag.reranker import Reranker
from data_pipeline.ingestion.ingest_qr_technique import (
    build_bm25_index,
    tokenize,
    vectorize_bm25,
)


@dataclass
class RetrievalResult:
    answer_found: bool
    should_escalate: bool
    escalade_reason: str
    top_candidates: list
    top1_score: float
    query: str
    step_timings: dict[str, float] = field(default_factory=dict)
    # Exemple: {"retrieval_hybrid_ms": 150.5, "reranking_ms": 45.3, "total_ms": 195.8}


class Retriever:
    """
    Charge Embedder + Reranker + index BM25 (corpus qr_technique) une seule
    fois dans __init__, puis exécute pré-qualification scope → retrieval
    hybride → reranking → gate d'abstention (cf. retrieve()).
    """

    def __init__(self, ingestion_config_path: str = DEFAULT_INGESTION_CONFIG_PATH):
        self.settings = get_settings()

        self.embedder = Embedder()
        self.reranker = Reranker()

        texts, self.ingestion_config = build_reference_corpus(ingestion_config_path)
        self.bm25, self.vocab = build_bm25_index(texts)

        self.collection_name = self.ingestion_config["collection_name"]
        self.client = QdrantClient(url=self.settings.qdrant_url, timeout=30)

        # Comparaison scope insensible à la casse/espaces : un champ ticket
        # mal saisi ("assistance" au lieu de "Assistance") ne doit pas être
        # traité comme hors-scope alors qu'il désigne la même valeur.
        self._scope_modules_norm = {m.strip().lower() for m in self.settings.rag_scope_modules}
        self._scope_priorities_norm = {p.strip().lower() for p in self.settings.rag_scope_priorities}

    def retrieve(
        self,
        query: str,
        module: str | None = None,
        priorite: str | None = None,
        top_n: int = 3,
    ) -> RetrievalResult:
        start_total = time.perf_counter()
        step_timings: dict[str, float] = {}

        # Étape 0 — pré-qualification scope
        start_scope = time.perf_counter()
        if self.settings.rag_module_filter_enabled:
            if module is not None and module.strip().lower() not in self._scope_modules_norm:
                elapsed_scope = (time.perf_counter() - start_scope) * 1000
                step_timings["scope_check_ms"] = elapsed_scope
                step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
                return RetrievalResult(
                    answer_found=False,
                    should_escalate=True,
                    escalade_reason="module_hors_scope",
                    top_candidates=[],
                    top1_score=0.0,
                    query=query,
                    step_timings=step_timings,
                )
            if priorite is not None and priorite.strip().lower() not in self._scope_priorities_norm:
                elapsed_scope = (time.perf_counter() - start_scope) * 1000
                step_timings["scope_check_ms"] = elapsed_scope
                step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
                return RetrievalResult(
                    answer_found=False,
                    should_escalate=True,
                    escalade_reason="priorite_hors_scope",
                    top_candidates=[],
                    top1_score=0.0,
                    query=query,
                    step_timings=step_timings,
                )
        elapsed_scope = (time.perf_counter() - start_scope) * 1000
        step_timings["scope_check_ms"] = elapsed_scope

        # Étape 1 — retrieval hybride (premier étage)
        start_hybrid = time.perf_counter()
        dense_query = self.embedder.encode(query)
        sparse_query = vectorize_bm25(tokenize(query), self.bm25, self.vocab)

        dense_results, sparse_results = fetch_dense_sparse_candidates(
            self.client,
            self.collection_name,
            dense_query,
            sparse_query,
            limit=self.settings.rag_first_stage_k,
        )
        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        pool_ids = [point_id for point_id, _ in fused[: self.settings.rag_first_stage_k]]

        candidates: list[dict[str, Any]] = []
        if pool_ids:
            points = self.client.retrieve(
                collection_name=self.collection_name, ids=pool_ids, with_payload=True
            )
            payload_par_id = {str(p.id): p.payload for p in points}
            candidates = [
                {"id": point_id, "payload": payload_par_id[point_id]}
                for point_id in pool_ids
                if point_id in payload_par_id
            ]
        elapsed_hybrid = (time.perf_counter() - start_hybrid) * 1000
        step_timings["retrieval_hybrid_ms"] = elapsed_hybrid

        # Étape 2 — reranking
        start_rerank = time.perf_counter()
        candidats_rerankes = self.reranker.rerank(query, candidates, top_n=top_n)
        elapsed_rerank = (time.perf_counter() - start_rerank) * 1000
        step_timings["reranking_ms"] = elapsed_rerank

        # Étape 3 — gate d'abstention
        top1_score = candidats_rerankes[0]["reranker_score"] if candidats_rerankes else 0.0
        answer_found = top1_score >= self.settings.rag_confidence_threshold
        should_escalate = not answer_found
        escalade_reason = "score_insuffisant" if should_escalate else ""

        # Étape 4 — retour avec timings
        step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
        return RetrievalResult(
            answer_found=answer_found,
            should_escalate=should_escalate,
            escalade_reason=escalade_reason,
            top_candidates=candidats_rerankes,
            top1_score=top1_score,
            query=query,
            step_timings=step_timings,
        )

    def health_check(self) -> dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": "ok",
                "collection": self.collection_name,
                "points_count": info.points_count,
            }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
