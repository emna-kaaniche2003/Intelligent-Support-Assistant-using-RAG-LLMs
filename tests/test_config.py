import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.config import Settings


class TestRagScopeFilterDefaults:
    """Constantes RAG Scope Filter (config.py) : valeurs par défaut, sans lecture .env."""

    def test_rag_confidence_threshold_est_un_float_entre_0_et_1(self, monkeypatch):
        monkeypatch.delenv("RAG_CONFIDENCE_THRESHOLD", raising=False)
        settings = Settings(_env_file=None)
        assert isinstance(settings.rag_confidence_threshold, float)
        assert 0.0 <= settings.rag_confidence_threshold <= 1.0

    def test_rag_scope_modules_et_priorities_sont_des_listes_non_vides(self, monkeypatch):
        monkeypatch.delenv("RAG_SCOPE_MODULES", raising=False)
        monkeypatch.delenv("RAG_SCOPE_PRIORITIES", raising=False)
        settings = Settings(_env_file=None)
        assert isinstance(settings.rag_scope_modules, list)
        assert len(settings.rag_scope_modules) > 0
        assert isinstance(settings.rag_scope_priorities, list)
        assert len(settings.rag_scope_priorities) > 0
