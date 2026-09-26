import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.rag.reranker import Reranker, build_candidate_text, sigmoid


def make_candidate(id_, categorie="Devis", question="Q ?", reponse="R.", **extra):
    payload = {"categorie": categorie, "question": question, "reponse": reponse}
    payload.update(extra.pop("payload_overrides", {}))
    return {"id": id_, "payload": payload, **extra}


class TestBuildCandidateText:
    def test_formule_categorie_question_reponse(self):
        payload = {"categorie": "Devis", "question": "Comment faire ?", "reponse": "Voici comment."}
        result = build_candidate_text(payload)
        assert result == "Devis. Comment faire ?. Voici comment."

    def test_troncature_a_max_length_mots(self):
        payload = {"categorie": "C", "question": "Q", "reponse": " ".join(["mot"] * 600)}
        result = build_candidate_text(payload, max_length=512)
        assert len(result.split()) == 512

    def test_champs_manquants_ne_levent_pas(self):
        result = build_candidate_text({})
        assert result == ". . "


class TestSigmoid:
    def test_borne_dans_zero_un(self):
        assert 0.0 < sigmoid(-10) < 0.5
        assert 0.5 < sigmoid(10) < 1.0
        assert sigmoid(0) == pytest.approx(0.5)


class TestReranker:
    def _make_reranker_with_mock(self, raw_scores):
        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            mock_model = MagicMock()
            mock_model.predict.return_value = raw_scores
            mock_cls.return_value = mock_model
            reranker = Reranker()
        return reranker, mock_model

    def test_charge_le_modele_une_seule_fois(self):
        # scores bruts (non bornés) : le tri doit rester cohérent après sigmoïde
        reranker, mock_model = self._make_reranker_with_mock([-2.0, 3.0])
        candidates = [make_candidate("a"), make_candidate("b")]
        reranker.rerank("query", candidates, top_n=2)
        reranker.rerank("query", candidates, top_n=2)
        assert reranker._model is mock_model
        # predict() appelé à chaque rerank(), mais le modèle chargé une seule fois (1 seul CrossEncoder())

    def test_tri_par_score_decroissant(self):
        reranker, _ = self._make_reranker_with_mock([-1.0, 4.0, 0.5])
        candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
        result = reranker.rerank("query", candidates, top_n=3)
        assert [c["id"] for c in result] == ["b", "c", "a"]

    def test_scores_normalises_entre_zero_et_un(self):
        # CrossEncoder.predict() n'est pas borné : reranker_score doit toujours être dans [0, 1]
        reranker, _ = self._make_reranker_with_mock([-10.0, 0.0, 10.0])
        candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
        result = reranker.rerank("query", candidates, top_n=3)
        for c in result:
            assert 0.0 <= c["reranker_score"] <= 1.0

    def test_troncature_top_n(self):
        reranker, _ = self._make_reranker_with_mock([-1.0, 4.0, 0.5])
        candidates = [make_candidate("a"), make_candidate("b"), make_candidate("c")]
        result = reranker.rerank("query", candidates, top_n=1)
        assert len(result) == 1
        assert result[0]["id"] == "b"

    def test_liste_vide(self):
        reranker, mock_model = self._make_reranker_with_mock([])
        result = reranker.rerank("query", [], top_n=3)
        assert result == []
        mock_model.predict.assert_not_called()

    def test_enrichit_sans_perdre_le_payload(self):
        reranker, _ = self._make_reranker_with_mock([0.5])
        candidates = [make_candidate("a", categorie="Devis")]
        result = reranker.rerank("query", candidates, top_n=1)
        assert result[0]["payload"]["categorie"] == "Devis"
        assert result[0]["id"] == "a"
