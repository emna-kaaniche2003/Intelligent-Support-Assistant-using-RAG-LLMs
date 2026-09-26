import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import types

from ai_engine.agents.gemini_client import GeminiClient, GeminiUnavailableError


def _make_response(text="ok", finish_reason=types.FinishReason.STOP, with_candidate=True):
    """Construit une réponse mockée avec un candidat (finish_reason, .text)."""
    mock_response = MagicMock()
    mock_response.text = text
    if with_candidate:
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = finish_reason
        mock_response.candidates = [mock_candidate]
    else:
        mock_response.candidates = []
    return mock_response


def _make_client(generate_content_side_effect=None, generate_content_return=None):
    """
    Construit un GeminiClient avec google.genai/google.oauth2 entièrement
    mockés (aucun appel réseau). Retourne (client, mock_generate_content).
    """
    with patch("google.oauth2.service_account.Credentials.from_service_account_file") as mock_creds, \
         patch("google.genai.Client") as mock_client_cls:
        mock_creds.return_value = MagicMock()
        mock_genai_client = MagicMock()
        if generate_content_side_effect is not None:
            mock_genai_client.models.generate_content.side_effect = generate_content_side_effect
        else:
            mock_genai_client.models.generate_content.return_value = generate_content_return
        mock_client_cls.return_value = mock_genai_client

        client = GeminiClient()

    return client, mock_genai_client.models.generate_content


class TestGenerateNominal:
    def test_generate_retourne_le_texte(self):
        client, mock_generate_content = _make_client(
            generate_content_return=_make_response("réponse générée")
        )

        result = client.generate("quelle est la question ?")

        assert result == "réponse générée"
        mock_generate_content.assert_called_once()

    def test_generate_propage_temperature_et_system_instruction(self):
        client, mock_generate_content = _make_client(
            generate_content_return=_make_response("ok")
        )

        client.generate("contenu", system_instruction="tu es un agent support")

        call_kwargs = mock_generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.temperature == 0
        assert config.system_instruction == "tu es un agent support"
        assert config.max_output_tokens == 1024

    def test_generate_max_output_tokens_configurable(self):
        client, mock_generate_content = _make_client(
            generate_content_return=_make_response("ok")
        )

        client.generate("contenu", max_output_tokens=256)

        config = mock_generate_content.call_args.kwargs["config"]
        assert config.max_output_tokens == 256


class TestGenerateErreurs:
    def test_sdk_exception_leve_geminiunavailableerror(self):
        client, _ = _make_client(generate_content_side_effect=TimeoutError("timeout réseau"))

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    def test_quota_exception_leve_geminiunavailableerror(self):
        client, _ = _make_client(generate_content_side_effect=RuntimeError("quota exceeded"))

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    def test_auth_exception_leve_geminiunavailableerror(self):
        client, _ = _make_client(generate_content_side_effect=PermissionError("auth invalide"))

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    def test_reponse_vide_leve_geminiunavailableerror(self):
        client, _ = _make_client(generate_content_return=_make_response(text=""))

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    def test_reponse_none_leve_geminiunavailableerror(self):
        client, _ = _make_client(generate_content_return=_make_response(text=None))

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    def test_reponse_sans_candidat_leve_geminiunavailableerror(self):
        client, _ = _make_client(
            generate_content_return=_make_response(with_candidate=False)
        )

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    def test_acces_text_qui_leve_est_capture_en_geminiunavailableerror(self):
        """
        Cas réel SDK : réponse bloquée (safety/recitation) où .text lève une
        ValueError plutôt que de renvoyer une chaîne vide. Doit être capturé
        et re-levé en GeminiUnavailableError, pas propagé brut.
        """
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = types.FinishReason.STOP
        mock_response.candidates = [mock_candidate]
        type(mock_response).text = property(
            lambda self: (_ for _ in ()).throw(ValueError("no text in response"))
        )
        client, _ = _make_client(generate_content_return=mock_response)

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")

    @pytest.mark.parametrize(
        "finish_reason",
        [
            types.FinishReason.SAFETY,
            types.FinishReason.RECITATION,
            types.FinishReason.MAX_TOKENS,
        ],
    )
    def test_finish_reason_anormal_leve_geminiunavailableerror(self, finish_reason):
        """finish_reason != STOP doit lever, même si .text renvoie un contenu non vide."""
        client, _ = _make_client(
            generate_content_return=_make_response(text="contenu partiel", finish_reason=finish_reason)
        )

        with pytest.raises(GeminiUnavailableError):
            client.generate("contenu")


class TestInitErreurs:
    def test_erreur_init_credentials_leve_geminiunavailableerror(self):
        with patch(
            "google.oauth2.service_account.Credentials.from_service_account_file",
            side_effect=FileNotFoundError("service account introuvable"),
        ):
            with pytest.raises(GeminiUnavailableError):
                GeminiClient()
