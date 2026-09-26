import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.api.main import app
from ai_engine.api.schemas import AgentResponse, MetaSupport

TICKET_VALIDE = {
    "ticket_id": "T-1",
    "module": "Facturation",
    "priorite": "Assistance",
    "description": "Le total dépasse le reste, je suis bloqué.",
}


@pytest.fixture
def client():
    """
    TestClient SANS `with` : le protocole lifespan ASGI n'est alors jamais
    déclenché (pas de startup/shutdown), donc jamais de vrai Retriever()/
    GeminiClient() instancié — zéro réseau, zéro chargement de modèle.
    app.state.agent4 est donc posé manuellement par chaque test.
    """
    app.state.warmup_ok = True
    app.state.warmup_error = None
    return TestClient(app)


class TestHealth:
    def test_health_repond_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestProcessTicketValide:
    def test_ticket_valide_retourne_200_et_agent_response(self, client):
        agent4 = MagicMock()
        agent4.traiter.return_value = AgentResponse(
            ticket_id="T-1",
            status="reponse_a_valider",
            reponse_client="Voici la procédure à suivre.",
            meta_support=MetaSupport(
                confidence=0.70,
                sources=[],
                escalade_reason=None,
                model_used="gemini-2.5-flash-lite",
            ),
        )
        app.state.agent4 = agent4

        response = client.post("/process-ticket", json=TICKET_VALIDE)

        assert response.status_code == 200
        body = response.json()
        assert body["ticket_id"] == "T-1"
        assert body["status"] == "reponse_a_valider"
        assert body["reponse_client"] == "Voici la procédure à suivre."
        assert body["meta_support"]["model_used"] == "gemini-2.5-flash-lite"
        agent4.traiter.assert_called_once()

    def test_aucune_logique_metier_ticket_transmis_tel_quel_a_agent4(self, client):
        agent4 = MagicMock()
        agent4.traiter.return_value = AgentResponse(
            ticket_id="T-1",
            status="escalade",
            reponse_client=None,
            meta_support=MetaSupport(confidence=0.0, sources=[], escalade_reason="module_hors_scope"),
        )
        app.state.agent4 = agent4

        client.post("/process-ticket", json=TICKET_VALIDE)

        ticket_recu = agent4.traiter.call_args.args[0]
        assert ticket_recu.ticket_id == TICKET_VALIDE["ticket_id"]
        assert ticket_recu.description == TICKET_VALIDE["description"]


class TestProcessTicketInvalide:
    def test_description_vide_retourne_422(self, client):
        agent4 = MagicMock()
        app.state.agent4 = agent4

        ticket_invalide = {**TICKET_VALIDE, "description": ""}
        response = client.post("/process-ticket", json=ticket_invalide)

        assert response.status_code == 422
        agent4.traiter.assert_not_called()

    def test_champ_manquant_retourne_422(self, client):
        agent4 = MagicMock()
        app.state.agent4 = agent4

        ticket_incomplet = {k: v for k, v in TICKET_VALIDE.items() if k != "module"}
        response = client.post("/process-ticket", json=ticket_incomplet)

        assert response.status_code == 422
        agent4.traiter.assert_not_called()


class TestProcessTicketErreurTechnique:
    def test_exception_inattendue_donne_200_avec_escalade_erreur_technique(self, client):
        agent4 = MagicMock()
        agent4.traiter.side_effect = RuntimeError("panne interne imprévue")
        app.state.agent4 = agent4

        response = client.post("/process-ticket", json=TICKET_VALIDE)

        assert response.status_code == 200
        body = response.json()
        assert body["ticket_id"] == "T-1"
        assert body["status"] == "escalade"
        assert body["reponse_client"] is None
        assert body["meta_support"]["escalade_reason"] == "erreur_technique"
        assert body["meta_support"]["sources"] == []
        assert body["meta_support"]["model_used"] is None

    def test_message_exception_absent_de_la_reponse(self, client):
        """Le contenu de l'exception (pouvant embarquer de la PII) ne fuit jamais au client."""
        agent4 = MagicMock()
        agent4.traiter.side_effect = ValueError("Jean Exemple 00000000")
        app.state.agent4 = agent4

        response = client.post("/process-ticket", json=TICKET_VALIDE)

        assert "Jean Exemple" not in response.text
        assert "00000000" not in response.text
