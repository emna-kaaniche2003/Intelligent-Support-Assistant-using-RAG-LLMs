# Guide : Mesure de la durée de traitement des réclamations

## Vue d'ensemble

Le système de **timing** permet de mesurer la durée de chaque étape du traitement d'une réclamation (ticket) par l'IA. Les durées sont incluses dans la réponse API pour diagnostic et monitoring.

## Architecture du timing

Le timing est implémenté au 3 niveaux :

### 1. **API FastAPI** (`ai_engine/api/main.py`)
- **Métrique** : `processing_duration_ms` 
- **Mesure** : Durée totale depuis la réception de la requête POST jusqu'au retour de la réponse
- **Logging** : INFO log avec ticket_id, durée et status
- **Cas d'erreur** : Inclus dans le logging et la réponse escalade

```log
INFO - Ticket ticket_001 traité en 523.4 ms (status=reponse_a_valider)
ERROR - Erreur technique inattendue (TimeoutError) lors du traitement du ticket ticket_002 (après 2150.5 ms)
```

### 2. **Agent4** (`ai_engine/agents/agent4_trivial.py`)
- **Métriques** :
  - `retrieval_ms` : durée du call au retriever
  - `anonymisation_ms` : durée du nettoyage/anonymisation de la question
  - `gemini_generation_ms` : durée de l'appel à Gemini
  - `total_ms` : durée totale du traitement

- **Sources des timings** :
  - Timings du retriever sont importés automatiquement dans la réponse
  - Inclus dans `meta_support.step_timings` (liste de StepTiming)

### 3. **Retriever** (`ai_engine/rag/retriever.py`)
- **Métriques** :
  - `scope_check_ms` : pré-qualification module/priorité
  - `retrieval_hybrid_ms` : retrieval dense + sparse + fusion RRF
  - `reranking_ms` : reranking cross-encoder
  - `total_ms` : durée totale du retrieval

- **Retour** : dict `step_timings` inclus dans `RetrievalResult`

## Exemple de réponse avec timing

```json
{
  "ticket_id": "ticket_001",
  "status": "reponse_a_valider",
  "reponse_client": "La réponse générée par l'IA...",
  "meta_support": {
    "confidence": 0.75,
    "sources": [...],
    "escalade_reason": null,
    "model_used": "gemini-2.5-flash-lite",
    "processing_duration_ms": 523.42,
    "step_timings": [
      {
        "step_name": "scope_check",
        "duration_ms": 0.5
      },
      {
        "step_name": "retrieval_hybrid",
        "duration_ms": 145.3
      },
      {
        "step_name": "reranking",
        "duration_ms": 78.2
      },
      {
        "step_name": "retrieval_total",
        "duration_ms": 224.0
      },
      {
        "step_name": "anonymisation",
        "duration_ms": 12.5
      },
      {
        "step_name": "gemini_generation",
        "duration_ms": 286.92
      }
    ]
  }
}
```

## Interprétation des durées

| Étape | Durée typique | Points clés |
|---|---|---|
| **scope_check** | < 1 ms | Quasi-instantané (vérification locale) |
| **retrieval_hybrid** | 100-200 ms | Embedding dense + BM25 sparse + fusion RRF |
| **reranking** | 30-80 ms | Cross-encoder BAAI/bge-reranker-v2-m3 |
| **anonymisation** | 5-20 ms | NER spaCy pour masquage des noms |
| **gemini_generation** | 200-500 ms | Appel réseau à Vertex AI + génération |
| **Total** | 300-800 ms | Dépend surtout de Gemini (réseau + inférence) |

## Cas d'escalade et timing

**Cas 1 : Escalade par scope (module/priorité hors-scope)**
```json
{
  "status": "escalade",
  "meta_support": {
    "escalade_reason": "module_hors_scope",
    "processing_duration_ms": 0.8,
    "step_timings": [
      {"step_name": "scope_check", "duration_ms": 0.8}
    ]
  }
}
```
→ Seule la pré-qualification est mesurée

**Cas 2 : Escalade par score insuffisant**
```json
{
  "status": "escalade",
  "meta_support": {
    "escalade_reason": "score_insuffisant",
    "processing_duration_ms": 224.5,
    "step_timings": [
      {"step_name": "scope_check", "duration_ms": 0.5},
      {"step_name": "retrieval_hybrid", "duration_ms": 145.3},
      {"step_name": "reranking", "duration_ms": 78.7}
    ]
  }
}
```
→ Pas de génération Gemini, timer s'arrête au reranking

**Cas 3 : Escalade par service indisponible**
```json
{
  "status": "escalade",
  "meta_support": {
    "escalade_reason": "service_indisponible",
    "processing_duration_ms": 512.3,
    "step_timings": [
      {"step_name": "scope_check", "duration_ms": 0.5},
      {"step_name": "retrieval_hybrid", "duration_ms": 145.3},
      {"step_name": "reranking", "duration_ms": 78.2},
      {"step_name": "retrieval_total", "duration_ms": 224.0},
      {"step_name": "anonymisation", "duration_ms": 12.5},
      {"step_name": "gemini_generation", "duration_ms": 250.3}  // Timeout après 250ms
    ]
  }
}
```
→ Gemini s'est connecté mais timeout ou erreur au-delà de 250ms

## Optimisation et monitoring

### 🔴 Flags rouges (à investiguer)

| Durée | Cause probable | Action |
|---|---|---|
| `retrieval_hybrid > 400ms` | Requête Qdrant lente ou réseau | Vérifier latence Qdrant, indexation |
| `reranking > 150ms` | Cross-encoder surchargé ou modèle lent | Vérifier GPU si local, batch size |
| `gemini_generation > 800ms` | Timeout réseau ou modèle surchargé | Vérifier quota Vertex AI, temps d'inférence |
| `anonymisation > 50ms` | Chargement NER spaCy à la demande | Déjà lazy-loadé, vérifier logs |

### 📊 Dashboarding (recommandé)

Capturer les timings pour créer un dashboard de monitoring :

```python
# Exemple : extracteur de timings pour Prometheus/Grafana
def extract_timings_for_monitoring(response: AgentResponse):
    """Extrait les timings pour exposition métrique."""
    meta = response.meta_support
    return {
        "ticket_id": response.ticket_id,
        "status": response.status,
        "processing_duration_ms": meta.processing_duration_ms,
        "step_timings": {
            s.step_name: s.duration_ms 
            for s in meta.step_timings
        },
    }
```

## Notes d'implémentation

- **Précision** : `time.perf_counter()` pour clock monotone haute résolution, conversion en millisecondes
- **Overhead** : ~ 0.1-0.5 ms pour les mesures elles-mêmes (négligeable)
- **Thread-safety** : `perf_counter()` est thread-safe en CPython (mais API asynchrone à surveiller)
- **Timings imbriqués** : `retrieval_ms` > `retrieval_hybrid_ms + reranking_ms` (overhead de traitement)
- **Cas d'erreur en retrait** : timings partiels capturés jusqu'à l'exception (voir cas 3 ci-dessus)

## Schéma Pydantic (schemas.py)

```python
class StepTiming(BaseModel):
    """Durée d'une étape du pipeline de traitement (support-only)."""
    step_name: str  # Identifiant lisible : "scope_check", "retrieval_hybrid", etc.
    duration_ms: float  # Durée en millisecondes (flottant, précision 0.01ms)

class MetaSupport(BaseModel):
    # ... champs existants ...
    processing_duration_ms: float = Field(
        default=0.0,
        description="Durée totale de traitement en millisecondes"
    )
    step_timings: list[StepTiming] = Field(
        default_factory=list,
        description="Durée par étape du pipeline (diagnostic)"
    )
```

## Historique des changements

- **S6+** : Implémentation du timing complet (3 niveaux : API / Agent4 / Retriever)
- **Prochaine étape** : Intégration monitoring (Prometheus, Grafana, alertes sur seuils)
