"""
config.py — Configuration centralisée du moteur IA
Lit les variables depuis le fichier .env via pydantic-settings.
Toutes les parties du projet importent depuis ce fichier UNIQUEMENT.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """
    Toutes les variables d'environnement du projet.
    Pydantic valide les types et lève une erreur claire si une variable
    obligatoire est manquante — plus de KeyError silencieux.
    """

    # --- LLM (Google Vertex AI — service account) ---
    google_application_credentials: str = Field(
        description="Chemin vers le fichier JSON du service account Google "
        "(obligatoire, aucune valeur par défaut — échec explicite au boot si absent)"
    )
    gemini_project: str = Field(
        default="clinisys-ai-assistant",
        description="ID du projet Google Cloud"
    )
    gemini_location: str = Field(
        default="us-central1",
        description="Région Vertex AI (us-central1 recommandé pour Gemini)"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Modèle Gemini Vertex AI à utiliser"
    )

    # --- Qdrant ---
    qdrant_url: str = Field(
        description="URL du serveur Qdrant (obligatoire, aucune valeur par défaut — "
        "échec explicite au boot si absent ; ex. http://localhost:6333 en natif, "
        "http://qdrant:6333 en conteneur)"
    )
    qdrant_collection: str = Field(
        default="tickets_resolus",
        description="Nom de la collection Qdrant"
    )
    enable_tickets_resolus_collection: bool = Field(
        default=False,
        description=(
            "Active la collection tickets_resolus dans le retrieval actif. "
            "Désactivée par défaut : pipeline validé techniquement mais retiré "
            "du retrieval après audit (résolutions jugées trop génériques). "
            "Voir le pipeline FAQ (reclamations_seules) pour l'approche retenue."
        )
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        description="Modèle sentence-transformers pour les embeddings"
    )

    # --- Détection des réclamations redondantes (detect_reclamations_redondantes) ---
    redundancy_similarity_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Seuil de similarité cosinus (embeddings) au-dessus duquel deux "
            "réclamations d'un même bucket sont regroupées (composantes connexes). "
            "0.90 validé sur dry-run réel après nettoyage du bruit d'embedding : "
            "distribution bimodale, vrais doublons concentrés >= 0.90."
        )
    )
    redundancy_dryrun_floor: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Plancher d'affichage des paires dans le rapport --dry-run : seules "
            "les paires de cosinus >= ce plancher sont listées (pour observer la "
            "distribution des scores et choisir le seuil)."
        )
    )
    redundancy_message_truncate: int = Field(
        default=400,
        ge=1,
        description=(
            "Troncature (en caractères) de message_clean dans la formule "
            "d'embedding texte_pour_embedding (troncature explicite pour éviter "
            "une coupe silencieuse par le modèle)."
        )
    )
    redundancy_min_informative_tokens: int = Field(
        default=4,
        ge=0,
        description=(
            "Nombre minimal de tokens informatifs (texte d'embedding après "
            "retrait du bruit de politesse et des termes du bucket) requis pour "
            "qu'un ticket soit comparé sémantiquement aux autres. Sous ce seuil : "
            "exclu du clustering, sort en singleton (methode_fusion = "
            "'singleton_info_faible'). Calé sur inspection manuelle des tickets "
            "réels (9 tickets génuinement vides à N=4, ex. 'TQB', 'INTEGRA', "
            "'B[TEL]') — 5 aurait exclu à tort des cas encore informatifs."
        )
    )

    # --- Anonymisation des personnes (NER, cleaning_utils.anonymize_text) ---
    enable_person_anonymization: bool = Field(
        default=True,
        description=(
            "Active le masquage des noms de personnes (entités PER, via spaCy) "
            "dans anonymize_text(). Filet de sécurité actif par défaut (pas une "
            "fonctionnalité expérimentale opt-in) : à désactiver uniquement en "
            "connaissance de cause, ex. environnement où spaCy/le modèle est "
            "indisponible. Voir check_env.py pour la vérification bloquante associée."
        )
    )
    person_ner_model: str = Field(
        default="fr_core_news_sm",
        description=(
            "Modèle spaCy utilisé pour la détection des entités PERSON (label "
            "spaCy 'PER') dans anonymize_text()."
        )
    )

    # --- FastAPI ---
    # api_host/api_port retirés (S6, conteneurisation) : aucun code ne les lisait
    # (uvicorn est lancé directement en ligne de commande, cf. Dockerfile CMD).
    api_env: str = Field(default="development")

    # --- Agent 4 ---
    agent4_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Score minimum pour une résolution automatique"
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre de tickets similaires récupérés par Qdrant"
    )

    # RAG Scope Filter — décision 2026-07-22, seuil 0.60 validé sur golden_dataset_v3.json
    # (35 positifs recall@3=77.1%, 25 négatifs 0 faux positif au seuil 0.60)
    # Filtre module/priorité : garde-fou MVP jusqu'à implémentation Agent 2
    # Hypothèse documentée : champ module du ticket supposé fiable en entrée
    rag_module_filter_enabled: bool = Field(
        default=True,
        description="Active/désactive le filtre de pré-qualification module/priorité (permet de le bypasser en test)"
    )
    rag_scope_modules: list[str] = Field(
        default=["Facturation (FW1)", "Facturation"],
        description="Modules de ticket acceptés pour une réponse automatique du RAG"
    )
    rag_scope_priorities: list[str] = Field(
        default=["Assistance", "Formation"],
        description="Priorités de ticket acceptées pour une réponse automatique du RAG"
    )
    rag_confidence_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Seuil du gate d'abstention : top1_reranker_score minimum pour répondre automatiquement"
    )
    rag_autosend_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Seuil d'envoi automatique sans validation support (status=reponse_auto). "
        "Dormant : aucun cas n'atteint 0.80 sur les données actuelles (golden_dataset_v3.json), "
        "branche présente et lue depuis config pour activation future sans redéploiement de code."
    )
    rag_first_stage_k: int = Field(
        default=20,
        ge=1,
        description="Taille du pool hybride (dense+sparse) remonté avant reranking"
    )

    @field_validator("rag_scope_modules", "rag_scope_priorities", mode="before")
    @classmethod
    def _split_comma_separated_env(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Retourne l'instance de configuration (singleton via cache).
    Usage :
        from ai_engine.config import get_settings
        settings = get_settings()
        print(settings.gemini_project)
    """
    return Settings()
