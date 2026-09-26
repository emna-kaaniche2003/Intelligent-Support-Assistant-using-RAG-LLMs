import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.agents.agent4_trivial import (
    PROMPT_PATH,
    SENTINEL_ACTION_INTERDITE,
    SENTINEL_FICHES_INSUFFISANTES,
    Agent4,
    format_context,
    map_sources,
)
from ai_engine.agents.gemini_client import GeminiUnavailableError
from ai_engine.api.schemas import TicketInput
from ai_engine.rag.retriever import RetrievalResult

CANDIDATE_1 = {
    "id": "uuid-aaa-111",
    "payload": {
        "categorie": "Assurances",
        "question": "Pourquoi le total dépasse le reste ?",
        "reponse": "Réduisez les pourcentages saisis pour que le total soit inférieur au reste.",
    },
    "reranker_score": 0.91,
}
CANDIDATE_2 = {
    "id": "uuid-bbb-222",
    "payload": {
        "categorie": "Encaissement",
        "question": "Comment annuler un encaissement ?",
        "reponse": "Ouvrez le dossier puis cliquez sur Annuler encaissement.",
    },
    "reranker_score": 0.75,
}

TICKET = TicketInput(
    ticket_id="T-1",
    module="Facturation",
    priorite="Assistance",
    description="Le total dépasse le reste, je suis bloqué.",
)


def _make_settings(rag_autosend_threshold=0.80, gemini_model="gemini-2.5-flash-lite"):
    settings = MagicMock()
    settings.rag_autosend_threshold = rag_autosend_threshold
    settings.gemini_model = gemini_model
    return settings


def _make_agent(retriever=None, gemini_client=None, settings=None):
    with patch(
        "ai_engine.agents.agent4_trivial.get_settings",
        return_value=settings or _make_settings(),
    ):
        agent = Agent4(retriever or MagicMock(), gemini_client or MagicMock())
    return agent


def _patch_anonymisation(identite=True):
    """
    Patch clean_html/anonymize_text au niveau du module agent4_trivial pour
    isoler les tests de la vraie chaîne de nettoyage/NER spaCy (mocks complets,
    zéro réseau, zéro chargement de modèle). Par défaut : passe-plat (identité),
    pour ne pas changer le comportement des tests qui ne portent pas sur
    l'anonymisation elle-même.
    """
    clean_html_mock = MagicMock(side_effect=(lambda t: t) if identite else None)
    anonymize_text_mock = MagicMock(side_effect=(lambda t: t) if identite else None)
    return patch.multiple(
        "ai_engine.agents.agent4_trivial",
        clean_html=clean_html_mock,
        anonymize_text=anonymize_text_mock,
    )


class TestFormatContext:
    def test_liste_vide_retourne_chaine_vide(self):
        assert format_context([]) == ""

    def test_formate_categorie_question_reponse_completes(self):
        texte = format_context([CANDIDATE_1])
        assert "Assurances" in texte
        assert "Pourquoi le total dépasse le reste ?" in texte
        assert "Réduisez les pourcentages saisis pour que le total soit inférieur au reste." in texte

    def test_delimiteur_clair_entre_plusieurs_fiches(self):
        texte = format_context([CANDIDATE_1, CANDIDATE_2])
        assert texte.index("Assurances") < texte.index("Encaissement")
        assert "---" in texte

    def test_ne_tronque_pas_la_reponse(self):
        reponse_longue = "x" * 1000
        candidat = {"id": "u1", "payload": {"categorie": "C", "question": "Q", "reponse": reponse_longue}}
        assert reponse_longue in format_context([candidat])


class TestMapSources:
    def test_liste_vide_retourne_liste_vide(self):
        assert map_sources([]) == []

    def test_mappe_point_id_categorie_question(self):
        sources = map_sources([CANDIDATE_1])
        assert len(sources) == 1
        source = sources[0]
        assert source.point_id == "uuid-aaa-111"
        assert source.id_reference is None
        assert source.categorie == "Assurances"
        assert source.question == "Pourquoi le total dépasse le reste ?"

    def test_mappe_plusieurs_candidats_dans_l_ordre(self):
        sources = map_sources([CANDIDATE_1, CANDIDATE_2])
        assert [s.point_id for s in sources] == ["uuid-aaa-111", "uuid-bbb-222"]
        assert all(s.id_reference is None for s in sources)

    def test_mappe_reponse_quand_presente_dans_payload(self):
        """Vérifie que le champ 'reponse' est copié du payload vers SourceFiche."""
        sources = map_sources([CANDIDATE_1])
        assert len(sources) == 1
        source = sources[0]
        assert source.reponse == "Réduisez les pourcentages saisis pour que le total soit inférieur au reste."
        # Vérifier aussi sur un second candidat
        sources = map_sources([CANDIDATE_2])
        assert sources[0].reponse == "Ouvrez le dossier puis cliquez sur Annuler encaissement."

    def test_mappe_reponse_none_quand_absente_du_payload(self):
        """Vérifie que reponse vaut None si le champ est absent du payload (pas de KeyError)."""
        candidat_sans_reponse = {
            "id": "uuid-ccc-333",
            "payload": {
                "categorie": "Facturation",
                "question": "Comment valider une facture ?",
                # 'reponse' est intentionnellement absent
            },
            "reranker_score": 0.80,
        }
        sources = map_sources([candidat_sans_reponse])
        assert len(sources) == 1
        assert sources[0].reponse is None  # .get("reponse") doit retourner None, pas lever d'exception


class TestAgent4EscaladeScope:
    def test_should_escalate_module_hors_scope_sans_appel_gemini(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=False,
            should_escalate=True,
            escalade_reason="module_hors_scope",
            top_candidates=[],
            top1_score=0.0,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        response = agent.traiter(TICKET)

        assert response.status == "escalade"
        assert response.reponse_client is None
        assert response.meta_support.escalade_reason == "module_hors_scope"
        assert response.meta_support.sources == []
        assert response.meta_support.model_used is None
        gemini_client.generate.assert_not_called()


class TestAgent4EscaladeScoreInsuffisant:
    def test_score_insuffisant_avec_sources_non_vides(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=False,
            should_escalate=True,
            escalade_reason="score_insuffisant",
            top_candidates=[CANDIDATE_1, CANDIDATE_2],
            top1_score=0.45,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        response = agent.traiter(TICKET)

        assert response.status == "escalade"
        assert response.meta_support.escalade_reason == "score_insuffisant"
        assert len(response.meta_support.sources) == 2
        assert response.meta_support.confidence == 0.45
        gemini_client.generate.assert_not_called()


class TestAgent4Sentinelle:
    def test_sentinelle_fiches_insuffisantes_leve_escalade_fiches_non_pertinentes(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.70,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = f"  {SENTINEL_FICHES_INSUFFISANTES}  "
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        with _patch_anonymisation():
            response = agent.traiter(TICKET)

        assert response.status == "escalade"
        assert response.reponse_client is None
        assert response.meta_support.escalade_reason == "fiches_non_pertinentes"
        assert len(response.meta_support.sources) == 1
        assert response.meta_support.model_used is None

    def test_sentinelle_action_interdite_leve_escalade_action_interdite(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.70,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = f"  {SENTINEL_ACTION_INTERDITE}  "
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        with _patch_anonymisation():
            response = agent.traiter(TICKET)

        assert response.status == "escalade"
        assert response.reponse_client is None
        assert response.meta_support.escalade_reason == "action_interdite"
        assert len(response.meta_support.sources) == 1
        assert response.meta_support.model_used is None


class TestAgent4GeminiUnavailable:
    def test_gemini_unavailable_leve_escalade_service_indisponible(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.70,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.side_effect = GeminiUnavailableError("timeout")
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        with _patch_anonymisation():
            response = agent.traiter(TICKET)

        assert response.status == "escalade"
        assert response.reponse_client is None
        assert response.meta_support.escalade_reason == "service_indisponible"
        assert response.meta_support.model_used is None
        assert len(response.meta_support.sources) == 1


class TestAgent4ReponseValide:
    def test_reponse_valide_donne_reponse_a_valider(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1, CANDIDATE_2],
            top1_score=0.70,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = "Voici la procédure à suivre."
        settings = _make_settings(rag_autosend_threshold=0.80, gemini_model="gemini-2.5-flash-lite")
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client, settings=settings)

        with _patch_anonymisation():
            response = agent.traiter(TICKET)

        assert response.status == "reponse_a_valider"
        assert response.reponse_client == "Voici la procédure à suivre."
        assert response.meta_support.model_used == "gemini-2.5-flash-lite"
        assert len(response.meta_support.sources) == 2
        assert response.meta_support.confidence == 0.70
        assert response.meta_support.escalade_reason is None

    def test_reponse_valide_ne_depasse_jamais_reponse_auto_sur_seuil_courant(self):
        """Seuil autosend (0.80) non atteint -> reste reponse_a_valider, jamais reponse_auto."""
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.79,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = "Réponse."
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        with _patch_anonymisation():
            response = agent.traiter(TICKET)

        assert response.status == "reponse_a_valider"

    def test_reponse_auto_dormante_activable_via_config(self):
        """Branche dormante : si le seuil configuré est franchi, status=reponse_auto."""
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.85,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = "Réponse."
        settings = _make_settings(rag_autosend_threshold=0.80)
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client, settings=settings)

        with _patch_anonymisation():
            response = agent.traiter(TICKET)

        assert response.status == "reponse_auto"

    def test_appelle_retrieve_avec_les_bons_arguments(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.70,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = "Réponse."
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        with _patch_anonymisation():
            agent.traiter(TICKET)

        retriever.retrieve.assert_called_once_with(
            TICKET.description, TICKET.module, TICKET.priorite, top_n=3
        )

    def test_appelle_gemini_avec_system_instruction_contenant_fiches_mais_pas_la_question(self):
        """
        system_instruction = rôle + règles + fiches UNIQUEMENT (jamais la question
        client, pour ne pas mélanger instruction/donnée ni dupliquer les tokens) ;
        la question anonymisée part exclusivement dans `contents`.
        """
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.70,
            query=TICKET.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = "Réponse."
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)

        with _patch_anonymisation():
            agent.traiter(TICKET)

        call_kwargs = gemini_client.generate.call_args.kwargs
        assert "Assurances" in call_kwargs["system_instruction"]
        assert TICKET.description not in call_kwargs["system_instruction"]
        assert call_kwargs["contents"] == TICKET.description


class TestAgent4Anonymisation:
    """
    Vérifie la séparation stricte des deux branches :
      - retrieval : description brute, jamais anonymisée (cf. Prompt A)
      - génération Gemini : description passée par clean_html() puis
        anonymize_text() avant injection dans `contents` UNIQUEMENT — le
        system_instruction (rôle + règles + fiches) ne contient jamais la
        question client, anonymisée ou non.
    Mocks complets sur clean_html/anonymize_text : zéro appel réseau, zéro
    chargement du modèle spaCy réel.
    """

    TICKET_PII = TicketInput(
        ticket_id="T-2",
        module="Facturation",
        priorite="Assistance",
        description="Bonjour, je suis Jean Exemple, mon numéro est 00000000.",
    )
    TEXTE_ANONYMISE = "Bonjour, je suis [PERSONNE], mon numéro est [TEL]."

    def _make_agent_reponse_valide(self, ticket):
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(
            answer_found=True,
            should_escalate=False,
            escalade_reason="",
            top_candidates=[CANDIDATE_1],
            top1_score=0.70,
            query=ticket.description,
        )
        gemini_client = MagicMock()
        gemini_client.generate.return_value = "Réponse."
        agent = _make_agent(retriever=retriever, gemini_client=gemini_client)
        return agent, retriever, gemini_client

    def test_retriever_recoit_la_description_non_anonymisee(self):
        agent, retriever, gemini_client = self._make_agent_reponse_valide(self.TICKET_PII)

        with patch.multiple(
            "ai_engine.agents.agent4_trivial",
            clean_html=MagicMock(return_value=self.TICKET_PII.description),
            anonymize_text=MagicMock(return_value=self.TEXTE_ANONYMISE),
        ):
            agent.traiter(self.TICKET_PII)

        retriever.retrieve.assert_called_once_with(
            self.TICKET_PII.description, self.TICKET_PII.module, self.TICKET_PII.priorite, top_n=3
        )
        assert "Jean Exemple" in retriever.retrieve.call_args.args[0]
        assert "00000000" in retriever.retrieve.call_args.args[0]

    def test_gemini_recoit_le_texte_anonymise_dans_contents_seul(self):
        agent, retriever, gemini_client = self._make_agent_reponse_valide(self.TICKET_PII)
        clean_html_mock = MagicMock(return_value=self.TICKET_PII.description)
        anonymize_text_mock = MagicMock(return_value=self.TEXTE_ANONYMISE)

        with patch.multiple(
            "ai_engine.agents.agent4_trivial",
            clean_html=clean_html_mock,
            anonymize_text=anonymize_text_mock,
        ):
            agent.traiter(self.TICKET_PII)

        clean_html_mock.assert_called_once_with(self.TICKET_PII.description)
        anonymize_text_mock.assert_called_once_with(self.TICKET_PII.description)

        call_kwargs = gemini_client.generate.call_args.kwargs
        # contents : la question anonymisée, seule
        assert call_kwargs["contents"] == self.TEXTE_ANONYMISE
        assert "Jean Exemple" not in call_kwargs["contents"]
        assert "00000000" not in call_kwargs["contents"]
        # system_instruction : ne contient plus la question du tout (ni brute ni anonymisée)
        assert self.TEXTE_ANONYMISE not in call_kwargs["system_instruction"]
        assert self.TICKET_PII.description not in call_kwargs["system_instruction"]
        assert "Jean Exemple" not in call_kwargs["system_instruction"]
        assert "00000000" not in call_kwargs["system_instruction"]

    def test_anonymisation_appelee_avec_clean_html_avant_anonymize_text(self):
        """anonymize_text() suppose un texte déjà nettoyé par clean_html() (ordre imposé)."""
        agent, retriever, gemini_client = self._make_agent_reponse_valide(self.TICKET_PII)
        appels = []
        clean_html_mock = MagicMock(side_effect=lambda t: appels.append("clean_html") or t)
        anonymize_text_mock = MagicMock(
            side_effect=lambda t: appels.append("anonymize_text") or self.TEXTE_ANONYMISE
        )

        with patch.multiple(
            "ai_engine.agents.agent4_trivial",
            clean_html=clean_html_mock,
            anonymize_text=anonymize_text_mock,
        ):
            agent.traiter(self.TICKET_PII)

        assert appels == ["clean_html", "anonymize_text"]


class TestPromptAntiDerive:
    def test_sentinelles_figurent_dans_le_prompt_systeme(self):
        contenu = PROMPT_PATH.read_text(encoding="utf-8")
        assert SENTINEL_ACTION_INTERDITE in contenu
        assert SENTINEL_FICHES_INSUFFISANTES in contenu
