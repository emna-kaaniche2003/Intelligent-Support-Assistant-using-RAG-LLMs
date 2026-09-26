import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.rag.retriever import Retriever, RetrievalResult


DEFAULT_SCOPE_MODULES = ["Facturation (FW1)", "Facturation"]
DEFAULT_SCOPE_PRIORITIES = ["Assistance", "Formation"]


def _make_settings(
    rag_module_filter_enabled=True,
    rag_confidence_threshold=0.60,
    rag_scope_modules=None,
    rag_scope_priorities=None,
    rag_first_stage_k=20,
):
    settings = MagicMock()
    settings.rag_module_filter_enabled = rag_module_filter_enabled
    settings.rag_confidence_threshold = rag_confidence_threshold
    settings.rag_scope_modules = rag_scope_modules or DEFAULT_SCOPE_MODULES
    settings.rag_scope_priorities = rag_scope_priorities or DEFAULT_SCOPE_PRIORITIES
    settings.rag_first_stage_k = rag_first_stage_k
    settings.qdrant_url = "http://localhost:6333"
    return settings


def _build_retriever(settings, stack):
    """
    Construit un Retriever avec toutes ses dépendances lourdes mockées
    (Embedder, Reranker, QdrantClient, reconstruction du corpus BM25) — aucun
    appel réseau ni chargement de modèle réel. Les patches sont enregistrés
    sur `stack` (contextlib.ExitStack fourni par l'appelant) pour rester actifs
    pendant retrieve(), pas seulement pendant __init__.
    """
    stack.enter_context(patch("ai_engine.rag.retriever.get_settings", return_value=settings))
    mock_embedder_cls = stack.enter_context(patch("ai_engine.rag.retriever.Embedder"))
    mock_reranker_cls = stack.enter_context(patch("ai_engine.rag.retriever.Reranker"))
    stack.enter_context(patch(
        "ai_engine.rag.retriever.build_reference_corpus",
        return_value=(["texte1", "texte2"], {"collection_name": "connaissances_modules"}),
    ))
    stack.enter_context(patch("ai_engine.rag.retriever.build_bm25_index", return_value=(MagicMock(), {})))
    mock_client_cls = stack.enter_context(patch("ai_engine.rag.retriever.QdrantClient"))

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = [0.1] * 768
    mock_embedder_cls.return_value = mock_embedder

    mock_reranker = MagicMock()
    mock_reranker_cls.return_value = mock_reranker

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    retriever = Retriever()
    return retriever, mock_embedder, mock_reranker, mock_client


def _make_retriever(settings=None, fused=None, stack=None):
    """
    Comme _build_retriever, mais patche en plus fetch_dense_sparse_candidates/
    reciprocal_rank_fusion (déjà testées indépendamment dans
    tests/test_retrieval.py) — `stack` doit rester ouvert pendant l'appel à
    retrieve() du test appelant.
    """
    settings = settings or _make_settings()
    fused = fused if fused is not None else []

    stack.enter_context(
        patch("ai_engine.rag.retriever.fetch_dense_sparse_candidates", return_value=([], []))
    )
    stack.enter_context(
        patch("ai_engine.rag.retriever.reciprocal_rank_fusion", return_value=fused)
    )
    return _build_retriever(settings, stack)


class TestPreQualification:
    def test_module_hors_scope_escalade(self):
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(stack=stack)
            result = retriever.retrieve("problème de paie", module="RH", priorite="Assistance")
        assert result.should_escalate is True
        assert result.escalade_reason == "module_hors_scope"
        assert result.answer_found is False
        mock_embedder.encode.assert_not_called()
        mock_reranker.rerank.assert_not_called()

    def test_priorite_hors_scope_escalade(self):
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(stack=stack)
            result = retriever.retrieve(
                "bug critique", module="Facturation", priorite="Coupure Système"
            )
        assert result.should_escalate is True
        assert result.escalade_reason == "priorite_hors_scope"
        mock_embedder.encode.assert_not_called()

    def test_priorite_casse_differente_passe_le_filtre(self):
        """'assistance' (minuscule) désigne la même priorité que 'Assistance' —
        ne doit PAS être traité comme hors-scope (comparaison insensible à la casse)."""
        fused = [("x", 0.9)]
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(
                fused=fused, stack=stack
            )
            mock_client.retrieve.return_value = [MagicMock(id="x", payload={"categorie": "Devis"})]
            mock_reranker.rerank.return_value = [
                {"id": "x", "payload": {"categorie": "Devis"}, "reranker_score": 0.70}
            ]
            result = retriever.retrieve(
                "pb affichage", module="Facturation", priorite="assistance"
            )
        assert result.escalade_reason != "priorite_hors_scope"
        mock_embedder.encode.assert_called_once()

    def test_module_casse_et_espaces_differents_passe_le_filtre(self):
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(stack=stack)
            mock_reranker.rerank.return_value = []
            result = retriever.retrieve(
                "question", module="  FACTURATION  ", priorite="Assistance"
            )
        assert result.escalade_reason != "module_hors_scope"
        mock_embedder.encode.assert_called_once()

    def test_filtre_desactive_execute_etape1_meme_hors_scope(self):
        settings = _make_settings(rag_module_filter_enabled=False)
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(
                settings=settings, stack=stack
            )
            mock_reranker.rerank.return_value = []
            result = retriever.retrieve("problème de paie", module="RH", priorite="Assistance")
        mock_embedder.encode.assert_called_once()
        assert result.escalade_reason != "module_hors_scope"


class TestGateAbstention:
    def test_top1_score_070_answer_found(self):
        fused = [("x", 0.9)]
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(
                fused=fused, stack=stack
            )
            mock_client.retrieve.return_value = [MagicMock(id="x", payload={"categorie": "Devis"})]
            mock_reranker.rerank.return_value = [
                {"id": "x", "payload": {"categorie": "Devis"}, "reranker_score": 0.70}
            ]
            result = retriever.retrieve(
                "je veux passer une partie sur la mutuelle",
                module="Facturation", priorite="Assistance",
            )
        assert result.answer_found is True
        assert result.should_escalate is False
        assert result.escalade_reason == ""
        assert result.top1_score == pytest.approx(0.70)

    def test_top1_score_050_score_insuffisant(self):
        fused = [("x", 0.9)]
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(
                fused=fused, stack=stack
            )
            mock_client.retrieve.return_value = [MagicMock(id="x", payload={"categorie": "Devis"})]
            mock_reranker.rerank.return_value = [
                {"id": "x", "payload": {"categorie": "Devis"}, "reranker_score": 0.50}
            ]
            result = retriever.retrieve(
                "requête hors-scope",
                module="Facturation", priorite="Assistance",
            )
        assert result.answer_found is False
        assert result.should_escalate is True
        assert result.escalade_reason == "score_insuffisant"


class TestRetrieveReturnsFullResult:
    def test_tous_les_champs_attendus(self):
        fused = [("x", 0.9)]
        with ExitStack() as stack:
            retriever, mock_embedder, mock_reranker, mock_client = _make_retriever(
                fused=fused, stack=stack
            )
            mock_client.retrieve.return_value = [MagicMock(id="x", payload={"categorie": "Devis"})]
            mock_reranker.rerank.return_value = [
                {"id": "x", "payload": {"categorie": "Devis"}, "reranker_score": 0.70}
            ]
            result = retriever.retrieve(
                "je veux passer une partie sur la mutuelle",
                module="Facturation", priorite="Assistance",
            )
            expected_candidates = mock_reranker.rerank.return_value
        assert isinstance(result, RetrievalResult)
        assert result.query == "je veux passer une partie sur la mutuelle"
        assert result.top_candidates == expected_candidates
        assert isinstance(result.answer_found, bool)
        assert isinstance(result.should_escalate, bool)
        assert isinstance(result.escalade_reason, str)
        assert isinstance(result.top1_score, float)
