"""
ai_engine/api/main.py
FastAPI de production. Lifespan : Retriever/GeminiClient/Agent4 instanciés
UNE SEULE FOIS au démarrage (chargement embedder + reranker + BM25 + client
Vertex, plusieurs secondes), rangés dans app.state — jamais réinstanciés par
requête. Endpoints minces : aucune décision métier ici, tout vit dans Agent4.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response

from ai_engine.agents.agent4_trivial import Agent4
from ai_engine.agents.gemini_client import GeminiClient
from ai_engine.api.schemas import AgentResponse, MetaSupport, TicketInput
from ai_engine.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# Ticket minimal pour le warmup : module/priorité dans le scope RAG pour
# exercer le pipeline complet (retrieval + Gemini), pas juste l'escalade
# scope. Le résultat (escalade ou pas) est jeté — seul l'effet de bord
# (cold-start payé) compte.
WARMUP_TICKET = TicketInput(
    ticket_id="warmup",
    module="Facturation",
    priorite="Assistance",
    description="warmup",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    retriever = Retriever()
    gemini_client = GeminiClient()
    app.state.agent4 = Agent4(retriever, gemini_client)

    # Warmup : paie le cold-start (auth Vertex AI + première inférence
    # embedder/reranker/Gemini) au boot plutôt qu'au premier ticket client
    # (mesure de référence : ~11s à chaud vs >30s à froid). Réutilise le
    # pipeline existant (agent4.traiter) — pas de duplication de logique.
    # GARDE-FOU : un warmup raté ne doit JAMAIS empêcher le démarrage ; au
    # runtime le pipeline dégradera vers ERREUR_TECHNIQUE (déjà géré par
    # process_ticket() ci-dessous), ce qui est le comportement voulu.
    start = time.monotonic()
    try:
        app.state.agent4.traiter(WARMUP_TICKET)
        elapsed = time.monotonic() - start
        logger.info("Warmup pipeline terminé en %.1fs", elapsed)
        app.state.warmup_ok = True
        app.state.warmup_error = None
    except Exception as exc:
        logger.warning("Warmup échoué (serveur démarré quand même) : %s", exc)
        app.state.warmup_ok = False
        app.state.warmup_error = f"{type(exc).__name__}: {exc}"

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health(request: Request, response: Response) -> dict[str, str]:
    # Aucun appel réseau ici : lit uniquement l'issue du warmup déjà exécuté
    # (app.state), stockée au lifespan — pas de nouvel appel Qdrant/Gemini.
    if not getattr(request.app.state, "warmup_ok", False):
        response.status_code = 503
        return {
            "status": "degraded",
            "detail": request.app.state.warmup_error or "warmup non exécuté",
        }
    return {"status": "ok"}


@app.post("/process-ticket", response_model=AgentResponse)
def process_ticket(ticket: TicketInput, request: Request) -> AgentResponse:
    agent4: Agent4 = request.app.state.agent4
    start = time.perf_counter()
    try:
        response = agent4.traiter(ticket)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "Ticket %s traité en %.1f ms (status=%s)",
            ticket.ticket_id,
            elapsed,
            response.status,
        )
        return response
    except Exception as exc:
        # Fail-safe : un ticket ne se perd jamais. On ne logge QUE le type
        # d'exception et le ticket_id — jamais str(exc)/traceback complet,
        # qui pourraient embarquer la réclamation client (PII) selon l'origine
        # de l'erreur (SDK Gemini, retriever...).
        elapsed = (time.perf_counter() - start) * 1000
        logger.error(
            "Erreur technique inattendue (%s) lors du traitement du ticket %s (après %.1f ms)",
            type(exc).__name__,
            ticket.ticket_id,
            elapsed,
        )
        return AgentResponse(
            ticket_id=ticket.ticket_id,
            status="escalade",
            reponse_client=None,
            meta_support=MetaSupport(
                confidence=0.0,
                sources=[],
                escalade_reason="erreur_technique",
                model_used=None,
                processing_duration_ms=elapsed,
            ),
        )
