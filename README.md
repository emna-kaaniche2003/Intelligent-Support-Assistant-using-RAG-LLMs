# Support IA — Système d'Intelligence Artificielle pour Automatisation des Tickets Support

## 📋 Objectif du Projet

Intégrer l'Intelligence Artificielle dans la plateforme de support client pour **réduire la charge de travail de l'équipe support** via une architecture multi-agents orchestrée, avec **validation humaine obligatoire** sur toute réponse.

### Contexte métier

- **Plateforme** : SaaS médical + e-commerce 
- **Défis abordés** :
  - Tickets mal qualifiés
  - Demandes Formation qui auraient pu être autosuffisantes (documentation + vidéos disponibles)
  - Pas de pré-diagnostic automatique des bugs/anomalies
  - Temps moyen de réponse trop élevé

---

## 🎯 Vision : Pipeline IA à 4 Agents

### Architecture conceptuelle (phases 2+)

```
Entrée : Ticket audio/texte brut
  ↓
[Agent 1]  Transcription voix (derja tunisien) → texte structuré
  ↓
[Agent 2]  Qualification/correction module + type priorité
  ↓
[Agent 3]  Classement + vérification statut client (RAG paiements)
  ↓
[Agent 4]  Réponses guidées par RAG + validation humaine (MVP ✅)
  ↓
Sortie : Proposition IA en attente de validation support OU escalade
```

**Agents 1–3** : Conception établie. Déploiement post-MVP (hors périmètre MVP).

---

## ✅ Réalisé — Agent 4 : RAG + Génération Gemini

### Architecture technique

**Composants** :
1. **Retrieval hybride** : dense (sentence-transformers, 768d, Cosine) + sparse (BM25, IDF global)
   - Fusion : Reciprocal Rank Fusion (RRF), poids égal 0.5/0.5, validé par grid search
   - Collection Qdrant : `connaissances_modules` (775 points, qr_technique facturation)

2. **Reranking** : cross-encoder (BAAI/bge-reranker-v2-m3, via `sentence_transformers.CrossEncoder`)
   - Réordonne le pool hybride (top-20) avant gate d'abstention

3. **Gate d'abstention** : score reranker minimum (`RAG_CONFIDENCE_THRESHOLD=0.60`)
   - Si `top1_reranker_score < 0.60` → **escalade** (pas de réponse générée)
   - Sinon → génération Gemini

4. **Génération** : Gemini 2.5 Flash Lite (Vertex AI)
   - Anonymisation **stricte** avant envoi cloud : `anonymize_text(clean_html(description))`
   - Prompt système : rôle, contraintes anti-action, style, exemples (seul contexte technique)
   - Contenu : question anonymisée (PII masquées, noms de personnes via NER spaCy)

5. **Orchestration** : FastAPI `/process-ticket` webhook
   - Appel synchrone depuis support-core
   - Réponse : schéma `AgentResponse` (statut_ia, draft_reponse, fiches_json, top1_score)
   - Erreur : `ERREUR_TECHNIQUE` visible côté support, jamais silencieuse

### Métriques d'évaluation (golden_dataset_v3.json, 60 questions, 35 positifs)

**Après reranking** :

| Métrique | Valeur |
|---|---|
| **Recall@1** | 62.9% |
| **Recall@3** | 77.1% |
| **Recall@5** | 80.0% |
| **Precision@1** | 62.9% |
| **MRR** | 0.702 |

**Gate d'abstention** (25 négatifs, questions hors-scope) :
- `top1_reranker_score` médian : 0.501 | max : 0.598
- **Faux positifs au seuil 0.60** : 0/25 (0%)
- **Faux positifs au seuil 0.70** : 0/25 (0%)

### Données ingérées

**Collection `connaissances_modules`** (production, active) :
- Source : qr_technique facturation (775 questions/réponses)
- Format : pivot unifié (id UUID déterministe, texte_pour_embedding, payload enrichi)
- Hybrid search : dense 768d Cosine + sparse BM25
- Schéma payload : {module, source_type, categorie, question, reponse, mots_cles, ...}

**Collection `tickets_resolus`** (legacy, désactivée) :
- Retiré après audit : résolutions jugées trop génériques
- Flag : `ENABLE_TICKETS_RESOLUS_COLLECTION=false`
- Code/données : conservés en référence documentée

### Arborescence du code Agent 4

```
ai_engine/
├── agents/
│   └── agent4_trivial.py          # Agent 4 : retrieval → génération Gemini → AgentResponse
├── rag/
│   ├── retriever.py               # Hybrid search (dense+sparse, RRF, rerank, gate 0.60)
│   ├── embedder.py                # Wrapper sentence-transformers (singleton)
│   ├── reranker.py                # Cross-encoder (BAAI/bge-reranker-v2-m3)
│   ├── evaluate_retrieval.py      # Évaluation golden dataset (grid search, rerank, négatifs)
│   └── debug_retriever.py         # Instrumentation diagnostic (manuel)
├── api/
│   └── main.py                    # FastAPI : endpoint `/process-ticket`
├── prompts/
│   └── agent4_system.txt          # Prompt système versionné (rôle, contraintes, exemples)
└── config.py                      # Configuration centralisée (.env)
```

### Validations

**Tests unitaires collectés** : 253 tests (`pytest --collect-only`)

**Tests spécifiques Agent 4** :
- `test_agent4_trivial.py` : mocks retriever/gemini_client, zéro réseau
- `test_retriever.py` : pré-qualification scope, gate abstention, reranking
- `test_retrieval.py` : golden dataset evals, grille RRF, rerank, résolution version v{N}

**Validation manuelle** (à date : 10/08/2026) :
- Ticket de démonstration (validation réussie) : succès simulé (top1_score=0.689, draft peuplé, statut A_VALIDER) ✅
- Ticket de démonstration (chemin d'erreur) : erreur réseau simulée (requête bloquée via DevTools, ERREUR_TECHNIQUE visible) ✅

---

## ✅ Réalisé — Intégration Plateforme

### Backend (support-core)

**Entité PropositionIA** :
- Champs : `statutIa` (EN_ATTENTE/EN_COURS/A_VALIDER/ESCALADE/ERREUR_TECHNIQUE)
- Champs : `statutValidation` (EN_ATTENTE/VALIDEE/MODIFIEE/REJETEE)
- Champs métier : `draftReponse`, `fichesJson`, `top1Score`, `reponseFinale`, `validePar`, `valideLe`
- Storage : SQL Server  (table `proposition_ia`)

**Endpoints REST** (pas de contrôle de rôle MVP, escaladé post-MVP) :
- `GET /api/propositions-ia/{idMessage}` — lecture (JSON : existe/draft/fiches/score)
- `PUT /api/propositions-ia/valider/{idMessage}` — valider + envoyer via `MessageService.repondre()`
- `PUT /api/propositions-ia/rejeter/{idMessage}` — rejeter (pas d'effet externe)
- `PUT /api/propositions-ia/relancer/{idMessage}` — relancer depuis ERREUR_TECHNIQUE (async)

**Flux asynchrone** :
```
validerEtEnvoyer() / rejeter() — synchrone, met à jour statutValidation
relancer()
  ↓ [sauve statutIa=EN_COURS]
  ↓ [publie MessageATraiterParIAEvent]
  ↓ [@TransactionalEventListener(AFTER_COMMIT)]
  ↓ [@Async("aiTaskExecutor")]
  → traiterMessageAvecIA() [appel support-ia, re-ingestion à Qdrant]
```

### Frontend (support-web-react)

**Composant PropositionIAPanel**  :
- Emplacement : `src/Components/GestionDesDemandes/DemandeDetail.js` (intégration consolidée)
- Affichage : draft, fiches top-3, score pertinence, boutons Valider/Rejeter/Relancer
- Redux : `PropositionIAReducer`, actions LOADING/GET/ERROR/RESET
- Actions : Valider / Rejeter / Relancer (appels PUT, pas de logique métier front)

**Validations** :
- Bleed au remount (close→reopen) : action `PROPOSITION_IA_RESET` corrigée ✅
- PropTypes `idMessage` : number (pas string) ✅
- Séquences Redux validées par DevTools (124279 = validation réussie, 124280 = chemin d'erreur)

### Scope et Pre-qualification

**Filtre de scope RAG** (guard MVP jusqu'à Agent 2) :
- Module acceptés : `Facturation (FW1)`, `Facturation`
- Priorités acceptées : `Assistance`, `Formation`
- Hors-scope → escalade immédiate (pas d'appel RAG/Gemini)

---

## 🛡️ Validation Humaine : Invariant Fondateur

### Garantie R1 — Aucun envoi sans validation

**Statuts de flux** :
```
[Réception ticket] → AI processing (async) → [A_VALIDER]
                                               ↓
                                      [Validation support]
                                        ↙      ↓      ↘
                                    [VALIDEE] [MODIFIEE] [REJETEE]
                                       ↓
                                    [Envoi client]
```

**Invariants de code** :
- Seule `validerEtEnvoyer()` peut appeler `messageService.repondre()` (mail au client)
- Préalable : `statutIa == A_VALIDER && statutValidation == EN_ATTENTE`
- Ordre impératif : `repondre()` d'abord, puis mise à jour PropositionIA (transaction atomique)
- Autre endpoint ? (`rejeter`, `relancer`) → aucun effet client externe

**Preuve runtime** :
- Ticket de démonstration (validation réussie) : statut A_VALIDER en base, draft peuplé, EN_ATTENTE
- Aucune mail client générée tant que validation=EN_ATTENTE

### Statut Autosend (R2 — Dormant et Prouvé Inatteignable)

**Configuration** :
```
RAG_AUTOSEND_THRESHOLD=0.80  # Dans config.py
```

**Réalité sur golden_dataset_v3.json (60 questions, 35 positifs)** :
- Distribution `top1_reranker_score` (bonnes réponses) : médiane ~0.72, max ~0.77
- Aucun cas ≥ 0.80 sur les données actuelles

**Statut branche code** :
- Present : `if top1_score >= AUTOSEND_THRESHOLD` dans `agent4_trivial.py`
- Executed : jamais (condition non satisfaite)

**Pourquoi non activé MVP** :
- Chevauchement de distributions : score d'une réponse correcte (~0.72 max) vs incorrect (~0.598 max)
- Séparation nette au seuil 0.60 (gate d'abstention) : 0% faux positif prouvé
- Seuil 0.80 non justifié par les données : aucune calibration métier

### Score de Pertinence (R3 — Pas une Confiance)

**Notation** :
- Clé JSON : `top1_reranker_score` (pas "confidence", pas "qualité")
- Source : sortie sigmoïde du cross-encoder (valeur [0, 1])
- **Non calibrée** : n'est PAS un pourcentage de justesse
- Usage front : affichage numérique brut, jamais en pourcentage
- Sémantique : "score de reranking", jamais "confiance" ou "pertinence en %"

**Affichage front** :
```json
{
  "propositionIA": {
    "score": 0.689,  // Sigmoïde reranker, non calibrée
    "scoreLabel": "pertinence"  // Clé UI pour styling
  }
}
```

---

## 📊 Roadmap par État

### ✅ RÉALISÉ

- **Agent 4** : Architecture RAG complète (dense+BM25, rerank, gate 0.60, Gemini)
- **Intégration backend** : PropositionIA CRUD, endpoints validation, flux async
- **Intégration frontend** : PropositionIAPanel, Redux, actions support
- **Validation manuelle** : Prouvée sur tickets réels, invariant R1 implémenté
- **Anonymisation** : Texte avant cloud, NER spaCy, défense en profondeur
- **Données** : 775 Q/R qr_technique facturation ingérées, hybrid search testé

### 🔄 EN COURS

- [MVP en validation support-core/support-web-react]

### 📋 DIFFÉRÉ POST-MVP

**Ré-ingestion optimisation** (5-10 pts recall gain) :
- UUID déterministe pour id_reference (tracabilité retrieval)
- Inclure mots_cles dans `texte_pour_embedding` (+10 pts recall@3)

**Agents 1–3 + Orchestrateur** (conception établie, implémentation ultérieure) :
- Agent 1 : transcription derja (risque linguistique élevé)
- Agent 2 : qualification module/priorité
- Agent 3 : classement + vérification client

**Clustering redondances FAQ** (phases 3+ pausées) :
- Phase 1-2 : extraction + filtrage retenues (réduction ~15% par clustering sémantique)
- Phase 3+ : génération questions canoniques LLM + ingestion connaissances_modules

**RBAC applicatif** (dette escaladée, bloque go-live E3.2) :
- Actuellement : aucun contrôle code sur endpoints (protection = réseau/déploiement)
- Requis : UserDetailsService réel, principal custom, règles `authorizeHttpRequests`
- Statut : non résolu au MVP (hors périmètre stage)

---

## 🏗️ Architecture du Projet

```
support-ia/                          # Racine du dépôt Git
│
├── data_pipeline/                   # Preprocessing + ingestion RAG
│   ├── queries/
│   │   ├── extract_tickets.sql     # Export SQL (réclamations + résolutions)
│   │   └── extract_tickets_faq.sql # Export allégé (réclamations seules)
│   ├── config/
│   │   ├── module_keywords.json    # Mots-clés par module (v0.2.0)
│   │   ├── qdrant_config.json      # Config collection
│   │   └── ingestion_config.json   # Config ingestion qr_technique
│   ├── raw/
│   │   ├── messages.csv            # [legacy] 111,561 tickets bruts
│   │   ├── messages_faq.csv        # [legacy] Réclamations seules
│   │   └── qr_technique__facturation__v1.json  # 775 Q/R facturation
│   ├── processed/
│   │   └── qr_technique__facturation__v1_pivot.json  # 775 Q/R pivot
│   ├── clean/
│   │   ├── reclamations_et_reponses.csv                    # [tickets_resolus] 2,230 tickets (référence)
│   │   ├── reclamations_seules.csv                        # [FAQ] Réclamations retenues
│   │   ├── reclamations_redondantes.csv                   # [FAQ] Clusters
│   │   └── reclamations_redondantes_membres.csv           # [FAQ] Détail clusters
│   ├── ingestion/
│   │   ├── create_collection.py    # Crée collection Qdrant
│   │   ├── ingest_qr_technique.py  # Ingère 775 Q/R
│   │   └── ingest_qdrant.py        # [legacy] Tickets résolus
│   ├── cleaning_utils.py           # Nettoyage HTML + anonymisation (NER spaCy)
│   ├── extract_and_filter.py       # [legacy] Nettoyage filtrage tickets
│   ├── extract_faq_source.py       # Nettoyage filtrage FAQ
│   └── detect_reclamations_redondantes.py  # Clustering sémantique
│
├── ai_engine/                       # Moteur IA
│   ├── agents/
│   │   └── agent4_trivial.py       # Agent 4 : MVP ✅
│   ├── rag/
│   │   ├── retriever.py            # Hybrid search + gate ✅
│   │   ├── embedder.py             # Embeddings ✅
│   │   ├── reranker.py             # Reranking ✅
│   │   └── evaluate_retrieval.py   # Évaluation golden dataset
│   ├── api/
│   │   └── main.py                 # FastAPI
│   ├── prompts/
│   │   └── agent4_system.txt       # Prompt système versionné
│   └── config.py                   # Configuration centralisée
│
├── tests/
│   ├── golden_dataset_v*.json      # Golden datasets (v1 historique, v3 actif)
│   ├── test_agent4_trivial.py      # Tests Agent 4
│   ├── test_retriever.py           # Tests retrieval
│   ├── test_retrieval.py           # Tests évaluation golden
│   └── ...                         # 253 tests collectés au total
│
├── docker-compose.yml              # Qdrant
├── .env                            # Configuration (JAMAIS commitée)
├── requirements.txt                # Dépendances Python 3.12
├── check_env.py                    # Vérification installation
├── CLAUDE.md                       # Documentation codebase
├── DECISIONS.md                    # Registre de décisions (référence)
└── README.md                       # Ce fichier
```

---

## 🚀 Installation & Démarrage

### 1. Prérequis

- **Python 3.12.9** (3.13+ non supporté par torch)
- **Docker** (Qdrant)
- **Git**
- **Visual C++ Redistributable 2022** (Windows)

### 2. Setup

```bash
# Cloner
git clone <repo> support-ia
cd support-ia

# Venv Python 3.12
py -3.12 -m venv venv312
# Linux/Mac : python3.12 -m venv venv312

# Activer
venv312\Scripts\activate
# Linux/Mac : source venv312/bin/activate

# Installer
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Lancer Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
# Vérifier : http://localhost:6333/dashboard
```

### 4. Configuration .env

```bash
# Copier .env.example → .env
GOOGLE_APPLICATION_CREDENTIALS=credentials/gemini_sa.json
GEMINI_PROJECT=
GEMINI_LOCATION=
GEMINI_MODEL=
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=connaissances_modules
EMBEDDING_MODEL=paraphrase-multilingual-mpnet-base-v2
ENABLE_PERSON_ANONYMIZATION=true
PERSON_NER_MODEL=fr_core_news_sm
```

> ⚠️ `credentials/gemini_sa.json` est gitignored. Copier `.example` et remplir avec le service account Google Cloud.

### 5. Vérifier installation

```bash
python check_env.py
# Vérifie : Python 3.12, .env, dépendances, Qdrant, Vertex AI
```

---

## 🛠️ Stack Technique

| Composant | Technologie | Raison |
|-----------|-------------|--------|
| **LLM** | Google Vertex AI — Gemini 2.5 Flash Lite | Multilingue, rapide, service account sécurisé |
| **Vector DB** | Qdrant 1.9.2 | Hybrid search natif (dense + BM25) |
| **Embeddings** | sentence-transformers 5.6.0 — paraphrase-multilingual-mpnet-base-v2 | Multilingue (FR + arabe), 768-dim |
| **Reranking** | BAAI/bge-reranker-v2-m3 | Cross-encoder, validation empirique |
| **API Gateway** | FastAPI 0.138.0 | Webhook depuis plateforme existante |
| **Conteneurisation** | Docker | Qdrant local/prod |
| **Anonymisation** | spaCy 3.8.14 — fr_core_news_sm | Masquage noms personnes (NER) |
| **Backend** | Spring Boot 3.2.2 (support-core) | Entité PropositionIA, endpoints, orchestration |
| **Frontend** | React 17.0.2 (support-web-react) | PropositionIAPanel, Redux, validation UI |

---

## 📝 Documentation & Références

- **Docs Vertex AI** : https://cloud.google.com/vertex-ai/docs
- **Docs Qdrant** : https://qdrant.tech/documentation/
- **Docs Gemini API** : https://googleapis.github.io/python-genai/

---

## ⚙️ Commandes Utiles

### Tests

```bash
# Collecter tous les tests
pytest --collect-only -q

# Lancer les tests (peut prendre plusieurs minutes)
pytest tests/

# Test d'un fichier spécifique
pytest tests/test_agent4_trivial.py -v
```

### Évaluation RAG

```bash
# Grid search pondération RRF
python ai_engine/rag/evaluate_retrieval.py --grid-search

# Évaluation avec reranking
python ai_engine/rag/evaluate_retrieval.py --rerank --first-stage-k 20 --top-n 3

# Diagnostic debug (manuel)
python ai_engine/rag/debug_retriever.py "question test" --module Facturation
```

### Ingestion données

```bash
# Créer collection Qdrant
python data_pipeline/ingestion/create_collection.py

# Ingérer qr_technique facturation (775 points)
python data_pipeline/ingestion/ingest_qr_technique.py --config data_pipeline/config/ingestion_config.json
```

--

**⚠️ Ceci est un MVP. Aucun envoi automatique n'est exécuté. Validation humaine obligatoire sur chaque réponse IA.**
