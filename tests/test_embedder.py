import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.rag.embedder import Embedder


def _make_embedder_with_mock(encode_return):
    with patch("ai_engine.rag.embedder.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 768
        mock_model.encode.return_value = encode_return
        mock_cls.return_value = mock_model
        embedder = Embedder()
    return embedder, mock_model, mock_cls


class TestEncode:
    def test_encode_retourne_liste_768(self):
        embedder, _, _ = _make_embedder_with_mock(np.zeros(768))
        result = embedder.encode("texte")
        assert isinstance(result, list)
        assert len(result) == 768


class TestEncodeBatch:
    def test_encode_batch_retourne_liste_de_listes_coherente(self):
        embedder, mock_model, _ = _make_embedder_with_mock(np.zeros((3, 768)))
        result = embedder.encode_batch(["a", "b", "c"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, list) and len(v) == 768 for v in result)
        mock_model.encode.assert_called_once()
        assert mock_model.encode.call_args.kwargs["batch_size"] == 3
        assert mock_model.encode.call_args.kwargs["normalize_embeddings"] is True


class TestSingleton:
    def test_modele_charge_une_seule_fois(self):
        with patch("ai_engine.rag.embedder.SentenceTransformer") as mock_cls:
            mock_model = MagicMock()
            mock_model.get_embedding_dimension.return_value = 768
            mock_model.encode.return_value = np.zeros(768)
            mock_cls.return_value = mock_model

            embedder = Embedder()
            embedder.encode("a")
            embedder.encode("b")
            embedder.encode_batch(["c", "d"])

            assert mock_cls.call_count == 1
