"""
ai_engine/agents/gemini_client.py
Wrapper autour de Vertex AI (google-genai) pour la génération de texte par
les agents — instancié une fois au démarrage, réutilisé (pas un singleton
au sens strict : l'unicité repose sur une instanciation unique par
l'appelant, pas garantie par le code). Config exclusivement via
ai_engine.config (aucune valeur en dur) — même pattern d'authentification
que data_pipeline/extract_and_filter.py (service account credentials).

Le wrapper ne décide JAMAIS d'escalade : toute erreur SDK (auth, réseau,
quota, timeout, réponse vide/bloquée par un safety filter) est capturée et
re-levée comme GeminiUnavailableError — c'est à l'appelant de trancher.
"""

from __future__ import annotations

from ai_engine.config import get_settings

DEFAULT_MAX_OUTPUT_TOKENS = 1024


class GeminiUnavailableError(Exception):
    """Levée quand Vertex AI/Gemini est indisponible ou renvoie une réponse inexploitable."""


class GeminiClient:
    """
    Charge un genai.Client (vertexai=True) à partir de ai_engine.config,
    puis expose generate() pour les appels de génération.
    """

    def __init__(self):
        from google import genai
        from google.oauth2 import service_account

        settings = get_settings()
        try:
            credentials = service_account.Credentials.from_service_account_file(
                settings.google_application_credentials,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._client = genai.Client(
                vertexai=True,
                project=settings.gemini_project,
                location=settings.gemini_location,
                credentials=credentials,
            )
        except Exception as exc:
            raise GeminiUnavailableError(
                f"Impossible d'initialiser le client Vertex AI : {exc}"
            ) from exc

        self._model = settings.gemini_model

    def generate(
        self,
        contents: str,
        system_instruction: str | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> str:
        """
        Appelle le modèle Gemini configuré (température 0, déterministe) et
        retourne le texte concaténé de la réponse.
        Lève GeminiUnavailableError sur toute erreur SDK ou réponse vide/bloquée.
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                raise GeminiUnavailableError(
                    f"Réponse Vertex AI sans candidat (modèle {self._model})"
                )

            finish_reason = candidates[0].finish_reason
            if finish_reason != types.FinishReason.STOP:
                raise GeminiUnavailableError(
                    f"Génération interrompue anormalement (modèle {self._model}, "
                    f"finish_reason={finish_reason})"
                )

            text = response.text
        except GeminiUnavailableError:
            raise
        except Exception as exc:
            raise GeminiUnavailableError(
                f"Échec de l'appel Vertex AI (modèle {self._model}) : {exc}"
            ) from exc

        if not text:
            raise GeminiUnavailableError(
                f"Réponse Vertex AI vide (modèle {self._model})"
            )
        return text
