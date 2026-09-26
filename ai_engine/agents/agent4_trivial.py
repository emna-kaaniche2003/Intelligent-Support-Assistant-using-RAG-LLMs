"""
ai_engine/agents/agent4_trivial.py
Agent 4 : étage de génération, responsabilité étroite TicketInput -> AgentResponse.
Consomme le retrieval déjà produit par ai_engine/rag/retriever.py (ne le refait pas).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ai_engine.agents.gemini_client import GeminiUnavailableError
from ai_engine.api.schemas import AgentResponse, MetaSupport, SourceFiche, TicketInput
from ai_engine.config import get_settings
from data_pipeline.cleaning_utils import anonymize_text, clean_html

SENTINEL_ACTION_INTERDITE = "ESCALADE_ACTION_INTERDITE"
SENTINEL_FICHES_INSUFFISANTES = "ESCALADE_FICHES_INSUFFISANTES"

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "agent4_system.txt"


def format_context(candidates: list[dict[str, Any]]) -> str:
    """
    Formate les fiches candidates pour {contexte_fiches} : catégorie + question +
    réponse complète (jamais tronquée), un délimiteur clair par fiche.
    Liste vide -> chaîne vide.
    """
    if not candidates:
        return ""

    blocs = []
    for candidate in candidates:
        payload = candidate["payload"]
        bloc = (
            f"Catégorie : {payload['categorie']}\n"
            f"Question : {payload['question']}\n"
            f"Réponse : {payload['reponse']}"
        )
        blocs.append(bloc)

    return "\n---\n".join(blocs)


def map_sources(candidates: list[dict[str, Any]]) -> list[SourceFiche]:
    """
    Mappe les candidats retrieval vers des SourceFiche (traçabilité support).
    id_reference reste None : le payload de production ne l'expose pas encore
    (cf. dette documentée sur SourceFiche). Liste vide -> liste vide.
    """
    return [
        SourceFiche(
            point_id=candidate["id"],
            id_reference=None,
            categorie=candidate["payload"]["categorie"],
            question=candidate["payload"]["question"],
            reponse=candidate["payload"].get("reponse"),
        )
        for candidate in candidates
    ]


class Agent4:
    """
    Reçoit retriever et gemini_client par injection (jamais instanciés en
    interne) pour permettre lifespan applicatif + mocks en test. Charge le
    prompt système une seule fois à la construction.
    """

    def __init__(self, retriever, gemini_client, prompt_path: Path = PROMPT_PATH):
        self.retriever = retriever
        self.gemini_client = gemini_client
        self.settings = get_settings()
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    def traiter(self, ticket: TicketInput) -> AgentResponse:
        start_total = time.perf_counter()
        step_timings: dict[str, float] = {}

        # Retrieval : TOUJOURS sur la description brute, jamais anonymisée
        # (cf. Prompt A) — le retriever fait son propre clean_html() en
        # interne, sans anonymisation. L'anonymisation ci-dessous est
        # strictement locale à la préparation du prompt Gemini.
        start_retrieval = time.perf_counter()
        result = self.retriever.retrieve(
            ticket.description, ticket.module, ticket.priorite, top_n=3
        )
        elapsed_retrieval = (time.perf_counter() - start_retrieval) * 1000
        step_timings["retrieval_ms"] = elapsed_retrieval

        # Capturer les timings du retriever (il les inclut dans step_timings)
        if result.step_timings:
            step_timings.update(result.step_timings)

        if result.should_escalate:
            step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
            return AgentResponse(
                ticket_id=ticket.ticket_id,
                status="escalade",
                reponse_client=None,
                meta_support=MetaSupport(
                    confidence=result.top1_score,
                    sources=map_sources(result.top_candidates),
                    escalade_reason=result.escalade_reason,
                    model_used=None,
                    processing_duration_ms=step_timings.get("total_ms", 0.0),
                    step_timings=self._build_step_timings_list(step_timings),
                ),
            )

        # Défense en profondeur adossée au warning déjà affiché côté client au
        # moment de la saisie (pas un remplacement) : la réclamation part vers
        # Gemini (Vertex AI, cloud) au moment de la génération, donc anonymisée
        # AVANT injection dans le prompt (noms -> [PERSONNE], téléphones ->
        # [TEL]). anonymize_text() suppose un texte déjà passé par clean_html()
        # (cf. data_pipeline/cleaning_utils.py). Limite assumée, PAS une
        # garantie : rappel spaCy imparfait sur les noms tunisiens, peu
        # représentés dans le corpus d'entraînement de fr_core_news_sm.
        start_anonymisation = time.perf_counter()
        question_anonymisee = anonymize_text(clean_html(ticket.description))
        elapsed_anonymisation = (time.perf_counter() - start_anonymisation) * 1000
        step_timings["anonymisation_ms"] = elapsed_anonymisation

        # system_instruction = rôle + règles + contexte (fiches) : jamais la
        # question client, pour ne pas mélanger instruction et donnée variable
        # ni dupliquer les tokens envoyés à Gemini. La question anonymisée
        # part uniquement dans `contents` (cf. appel generate() ci-dessous).
        contexte_fiches = format_context(result.top_candidates)
        prompt = self.system_prompt.format(contexte_fiches=contexte_fiches)
        sources = map_sources(result.top_candidates)

        try:
            start_gemini = time.perf_counter()
            texte = self.gemini_client.generate(
                contents=question_anonymisee, system_instruction=prompt
            )
            elapsed_gemini = (time.perf_counter() - start_gemini) * 1000
            step_timings["gemini_generation_ms"] = elapsed_gemini
        except GeminiUnavailableError:
            step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
            return AgentResponse(
                ticket_id=ticket.ticket_id,
                status="escalade",
                reponse_client=None,
                meta_support=MetaSupport(
                    confidence=result.top1_score,
                    sources=sources,
                    escalade_reason="service_indisponible",
                    model_used=None,
                    processing_duration_ms=step_timings.get("total_ms", 0.0),
                    step_timings=self._build_step_timings_list(step_timings),
                ),
            )

        texte_normalise = texte.strip()
        if texte_normalise in (SENTINEL_ACTION_INTERDITE, SENTINEL_FICHES_INSUFFISANTES):
            # Deux causes distinctes derrière ESCALADE_* : garde-fou anti-action
            # (demande explicite de modification de données) vs fiches
            # retrouvées mais ne couvrant pas réellement la question — cf.
            # agent4_system.txt (deux sentinelles dédiées, pas de logique
            # additionnelle ici, juste le mapping vers la raison support).
            escalade_reason = (
                "action_interdite"
                if texte_normalise == SENTINEL_ACTION_INTERDITE
                else "fiches_non_pertinentes"
            )
            step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
            return AgentResponse(
                ticket_id=ticket.ticket_id,
                status="escalade",
                reponse_client=None,
                meta_support=MetaSupport(
                    confidence=result.top1_score,
                    sources=sources,
                    escalade_reason=escalade_reason,
                    model_used=None,
                    processing_duration_ms=step_timings.get("total_ms", 0.0),
                    step_timings=self._build_step_timings_list(step_timings),
                ),
            )

        # reponse_auto reste dormant : aucun cas n'atteint rag_autosend_threshold
        # (0.80) sur les données actuelles, branche présente et lue depuis config.
        status = (
            "reponse_auto"
            if result.top1_score >= self.settings.rag_autosend_threshold
            else "reponse_a_valider"
        )

        step_timings["total_ms"] = (time.perf_counter() - start_total) * 1000
        return AgentResponse(
            ticket_id=ticket.ticket_id,
            status=status,
            reponse_client=texte,
            meta_support=MetaSupport(
                confidence=result.top1_score,
                sources=sources,
                escalade_reason=None,
                model_used=self.settings.gemini_model,
                processing_duration_ms=step_timings.get("total_ms", 0.0),
                step_timings=self._build_step_timings_list(step_timings),
            ),
        )

    def _build_step_timings_list(self, step_timings: dict[str, float]):
        """
        Convertit le dictionnaire de timings en liste de StepTiming.
        Exclut 'total_ms' et 'retrieval_hybrid_ms' du détail (ils sont redondants).
        """
        from ai_engine.api.schemas import StepTiming

        # Ordre logique des étapes à afficher
        step_order = [
            "scope_check_ms",
            "retrieval_hybrid_ms",
            "reranking_ms",
            "retrieval_ms",
            "anonymisation_ms",
            "gemini_generation_ms",
        ]

        timings_list = []
        for step_key in step_order:
            if step_key in step_timings:
                # Mapper les clés internes aux noms lisibles pour le support
                step_name_map = {
                    "scope_check_ms": "scope_check",
                    "retrieval_hybrid_ms": "retrieval_hybrid",
                    "reranking_ms": "reranking",
                    "retrieval_ms": "retrieval_total",
                    "anonymisation_ms": "anonymisation",
                    "gemini_generation_ms": "gemini_generation",
                }
                step_name = step_name_map.get(step_key, step_key.replace("_ms", ""))
                timings_list.append(
                    StepTiming(
                        step_name=step_name,
                        duration_ms=round(step_timings[step_key], 2),
                    )
                )

        return timings_list
