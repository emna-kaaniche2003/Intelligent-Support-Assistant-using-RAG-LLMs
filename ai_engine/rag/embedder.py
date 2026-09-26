"""
ai_engine/rag/embedder.py
Encodage dense des requêtes/textes pour le RAG (Agent 4). Même logique
d'encodage que data_pipeline/ingestion/ingest_qr_technique.py
(encode_dense_batch) et ai_engine/rag/evaluate_retrieval.py
(model.encode(..., normalize_embeddings=True)), extraite ici en composant
réutilisable de production — recette figée, NE PAS modifier
normalize_embeddings ni le modèle sans revalidation (cf. CLAUDE.md § RAG).
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from ai_engine.config import get_settings


class Embedder:
    """
    Charge le modèle sentence-transformers (settings.embedding_model) une
    seule fois en attribut d'instance, puis encode requêtes (encode) et
    corpus (encode_batch) avec normalize_embeddings=True.
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)
        dim = self._model.get_embedding_dimension()
        print(f"Embedder : modèle {self.model_name} chargé ({dim}d)")

    def encode(self, text: str) -> list[float]:
        """Encode un texte unique (requête entrante en production)."""
        vector = self._model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True
        )
        return vector.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode une liste de textes (utilisé par les scripts d'ingestion)."""
        vectors = self._model.encode(
            texts,
            batch_size=len(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]
