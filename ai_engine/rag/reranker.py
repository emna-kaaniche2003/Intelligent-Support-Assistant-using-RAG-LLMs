"""
ai_engine/rag/reranker.py
Second-étage de reranking par cross-encoder, appliqué sur le pool de candidats
déjà produit par le retrieval hybride de premier étage (fetch_dense_sparse_candidates
+ reciprocal_rank_fusion, ai_engine/rag/evaluate_retrieval.py).

Composant réutilisable, sans dépendance à Qdrant ni à la config de production :
prend une liste de candidats déjà enrichis de payload et retourne un sous-ensemble
trié par pertinence sémantique cross-encoder (BAAI/bge-reranker-v2-m3).

Backend — écart documenté avant écriture du code :
FlagEmbedding.FlagReranker (chemin nominal demandé) est installé (1.4.0, dernière
version disponible) mais lève `XLMRobertaTokenizer has no attribute
prepare_for_model` à l'exécution avec transformers==5.12.1 (pinné par le projet
pour sentence-transformers==5.6.0, cf. requirements.txt) — API interne retirée
en transformers 5.x, jamais corrigée côté FlagEmbedding (1.4.0 = dernière release
PyPI, pas de version plus récente disponible). Downgrader transformers globalement
mettrait en danger les embeddings de production (sentence-transformers). On
utilise donc le fallback explicitement autorisé : sentence_transformers.CrossEncoder
sur le même modèle BAAI/bge-reranker-v2-m3 (déjà une dépendance du projet, aucun
paquet supplémentaire), avec normalisation sigmoïde manuelle des scores dans
[0, 1] (CrossEncoder.predict() n'est pas borné, contrairement à
FlagReranker.compute_score(normalize=True)).
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LENGTH = 512


def build_candidate_text(payload: dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """"{categorie}. {question}. {reponse}" tronqué à ~max_length tokens (approximé par mots)."""
    categorie = payload.get("categorie") or ""
    question = payload.get("question") or ""
    reponse = payload.get("reponse") or ""
    texte = f"{categorie}. {question}. {reponse}"
    mots = texte.split()
    if len(mots) > max_length:
        texte = " ".join(mots[:max_length])
    return texte


class Reranker:
    """
    Charge BAAI/bge-reranker-v2-m3 (sentence_transformers.CrossEncoder, cf. note
    de backend en tête de module) une seule fois, puis réordonne des pools de
    candidats hybrides par cross-encoder.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_length: int = DEFAULT_MAX_LENGTH):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.max_length = max_length
        self._model = CrossEncoder(model_name, max_length=max_length)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """
        candidates : liste de dicts contenant au moins "id" + "payload" (payload avec
        au minimum categorie/question/reponse).
        Retourne les candidats triés par reranker_score décroissant, tronqués à top_n,
        chacun enrichi de la clé "reranker_score" (float dans [0, 1], sigmoïde du
        score brut CrossEncoder — non borné nativement).
        """
        if not candidates:
            return []

        pairs = [
            [query, build_candidate_text(candidate["payload"], self.max_length)]
            for candidate in candidates
        ]
        raw_scores = self._model.predict(pairs)

        scored = [
            {**candidate, "reranker_score": sigmoid(float(raw_score))}
            for candidate, raw_score in zip(candidates, raw_scores)
        ]
        scored.sort(key=lambda c: c["reranker_score"], reverse=True)
        return scored[:top_n]


def sigmoid(x: float) -> float:
    """Normalisation [0, 1] pour le fallback CrossEncoder (compute_score non borné)."""
    return 1.0 / (1.0 + math.exp(-x))
