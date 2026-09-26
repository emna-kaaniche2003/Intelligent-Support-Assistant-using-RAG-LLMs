"""
ai_engine/api/schemas.py
Modèles Pydantic v2 pour l'API de support-IA.
Schémas client-visible et support-only, sans logique métier.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """Données d'entrée d'un ticket client."""

    ticket_id: str = Field(
        ...,
        min_length=1,
        description="Identifiant unique du ticket (non vide, visible client).",
    )
    module: str = Field(
        ...,
        min_length=1,
        description="Module applicatif (non vide, ex: Facturation, visible client).",
    )
    priorite: str = Field(
        ...,
        min_length=1,
        description="Type de priorité (non vide, ex: Assistance, visible client).",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Contenu du ticket client (non vide, visible client).",
    )


class SourceFiche(BaseModel):
    """Traçabilité d'une fiche source RAG (support-only, jamais au client)."""

    point_id: str = Field(
        ...,
        description="Identifiant technique du point Qdrant (support-only) ; "
        "pas une référence métier lisible.",
    )
    id_reference: str | None = Field(
        default=None,
        description="Référence métier de la fiche (support-only) ; None tant que le "
        "payload n'est pas enrichi à l'ingestion — cf. dette documentée.",
    )
    categorie: str = Field(
        ...,
        description="Catégorie RAG de la fiche (support-only, ex: Encaissement).",
    )
    question: str = Field(
        ...,
        description="Question source de la fiche (support-only, texte de traçabilité).",
    )
    reponse: str | None = Field(
        default=None,
        description="Contenu de la réponse de la fiche (support-only, depuis payload Qdrant). "
        "None si absent du payload ou si la fiche n'en expose pas.",
    )


class StepTiming(BaseModel):
    """Durée d'une étape du pipeline de traitement (support-only)."""

    step_name: str = Field(
        ...,
        description="Nom de l'étape (ex: retrieval_dense, retrieval_sparse, "
        "reranking, gemini_generation, etc.).",
    )
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Durée en millisecondes (nombre flottant pour précision).",
    )


class MetaSupport(BaseModel):
    """Métadonnées de support (support-only, jamais visibles au client)."""

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="top1_score du reranker [0.0, 1.0] (support-only). "
        "Score heuristique, PAS une probabilité calibrée.",
    )
    sources: list[SourceFiche] = Field(
        default_factory=list,
        description="Fiches RAG remontées (support-only, traçabilité). "
        "Renseigné même en cas d'escalade, pour que le support voie ce que l'IA a trouvé.",
    )
    escalade_reason: str | None = Field(
        default=None,
        description="Raison de l'escalade si status=escalade (support-only). "
        "Valeurs: module_hors_scope | priorite_hors_scope | score_insuffisant | "
        "action_interdite | fiches_non_pertinentes | service_indisponible. "
        "None si une réponse est générée.",
    )
    model_used: str | None = Field(
        default=None,
        description="Modèle LLM utilisé (support-only, ex: gemini-2.5-flash-lite). "
        "None si escalade sans appel LLM.",
    )
    processing_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Durée totale de traitement en millisecondes (support-only). "
        "Mesurée du début du traitement jusqu'à la génération de la réponse.",
    )
    step_timings: list[StepTiming] = Field(
        default_factory=list,
        description="Durée par étape du pipeline (support-only, diagnostic). "
        "Permet d'identifier les goulots d'étranglement. Liste vide si pas de détail disponible.",
    )


class AgentResponse(BaseModel):
    """Réponse de l'agent RAG (client-visible + support-only)."""

    ticket_id: str = Field(
        ...,
        description="Identifiant du ticket (visible client).",
    )
    status: Literal["reponse_auto", "reponse_a_valider", "escalade"] = Field(
        ...,
        description="reponse_auto (envoi direct) / reponse_a_valider (attente support) "
        "/ escalade (aucune réponse générée). Détermine le comportement plateforme.",
    )
    reponse_client: str | None = Field(
        default=None,
        description="Texte de réponse en français (visible client). None si escalade.",
    )
    meta_support: MetaSupport = Field(
        ...,
        description="Métadonnées RAG/support (support-only, jamais au client).",
    )