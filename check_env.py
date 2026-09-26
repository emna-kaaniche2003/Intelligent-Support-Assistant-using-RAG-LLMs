"""
check_env.py — Vérification complète de l'environnement
Lancer depuis la racine : python check_env.py
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print("=" * 55)
print("  Support-IA — Vérification de l'environnement")
print("=" * 55)

errors = []

# 1. Version Python
print("\n[1/6] Version Python...")
version = sys.version_info
if version.major == 3 and version.minor == 12:
    print(f"  OK Python {version.major}.{version.minor}.{version.micro}")
else:
    msg = f"Python 3.12 requis, trouvé {version.major}.{version.minor}"
    print(f"  ERREUR {msg}")
    errors.append(msg)

# 2. Fichier .env + variables Vertex AI
print("\n[2/6] Fichier .env + configuration Vertex AI...")
try:
    from dotenv import dotenv_values
except ImportError:
    msg = "python-dotenv non installé — pip install python-dotenv"
    print(f"  ERREUR {msg}")
    errors.append(msg)
    dotenv_values = None

env_vals = {}
if os.path.exists(".env"):
    print("  OK .env trouvé")
    if dotenv_values:
        env_vals = dotenv_values(".env")
        sa_path = env_vals.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        project  = env_vals.get("GEMINI_PROJECT", "")
        location = env_vals.get("GEMINI_LOCATION", "")
        model    = env_vals.get("GEMINI_MODEL", "")

        if sa_path:
            print(f"  OK GOOGLE_APPLICATION_CREDENTIALS : {sa_path}")
        else:
            msg = "GOOGLE_APPLICATION_CREDENTIALS manquant dans .env"
            print(f"  ERREUR {msg}")
            errors.append(msg)

        if project:
            print(f"  OK GEMINI_PROJECT : {project}")
        else:
            msg = "GEMINI_PROJECT manquant dans .env"
            print(f"  ERREUR {msg}")
            errors.append(msg)

        if location:
            print(f"  OK GEMINI_LOCATION : {location}")
        else:
            print("  WARN GEMINI_LOCATION non défini — 'us-central1' utilisé par défaut")

        if model:
            print(f"  OK GEMINI_MODEL : {model}")
        else:
            print("  WARN GEMINI_MODEL non défini — valeur par défaut utilisée")
else:
    msg = ".env introuvable — copier .env.example → .env et remplir les valeurs"
    print(f"  ERREUR {msg}")
    errors.append(msg)

# 2b. Fichier service account JSON
print("\n[2b/6] Fichier service account JSON...")
sa_path = env_vals.get("GOOGLE_APPLICATION_CREDENTIALS", "credentials/gemini_sa.json")
if os.path.exists(sa_path):
    import json
    try:
        with open(sa_path, encoding="utf-8") as f:
            sa_data = json.load(f)
        sa_type    = sa_data.get("type", "?")
        sa_project = sa_data.get("project_id", "?")
        sa_email   = sa_data.get("client_email", "?")
        print(f"  OK {sa_path} — type={sa_type}, project={sa_project}")
        print(f"     email : {sa_email}")
    except Exception as e:
        msg = f"Erreur lecture {sa_path} : {e}"
        print(f"  ERREUR {msg}")
        errors.append(msg)
else:
    msg = f"Service account JSON introuvable : {sa_path}"
    print(f"  ERREUR {msg}")
    errors.append(msg)

# 3. Dépendances Python clés
print("\n[3/6] Dépendances Python...")
deps = {
    "google.genai":          "google-genai",
    "google.oauth2":         "google-auth",
    "qdrant_client":         "qdrant-client",
    "sentence_transformers": "sentence-transformers",
    "fastapi":               "fastapi",
    "pydantic_settings":     "pydantic-settings",
    "dotenv":                "python-dotenv",
}
for module, package in deps.items():
    try:
        __import__(module)
        print(f"  OK {package}")
    except ImportError:
        msg = f"{package} non installé — pip install {package}"
        print(f"  ERREUR {msg}")
        errors.append(msg)

# 4. Connexion Qdrant
print("\n[4/6] Connexion Qdrant...")
try:
    from qdrant_client import QdrantClient
    url = env_vals.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=3)
    client.get_collections()
    print(f"  OK Qdrant accessible sur {url}")
except Exception as e:
    print(f"  WARN Qdrant inaccessible — lancer : docker run -d --name qdrant -p 6333:6333 qdrant/qdrant")
    print(f"       Détail : {e}")

# 5. Test Vertex AI Gemini (appel réel)
print("\n[5/6] Test Vertex AI Gemini...")
try:
    from google import genai
    from google.oauth2 import service_account as sa_module

    sa_path  = env_vals.get("GOOGLE_APPLICATION_CREDENTIALS", "credentials/gemini_sa.json")
    project  = env_vals.get("GEMINI_PROJECT", "clinisys-ai-assistant")
    location = env_vals.get("GEMINI_LOCATION", "us-central1")
    model    = env_vals.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

    if not os.path.exists(sa_path):
        print(f"  SKIP service account JSON introuvable : {sa_path}")
    else:
        credentials = sa_module.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        vertex_client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )
        response = vertex_client.models.generate_content(
            model=model,
            contents="Réponds uniquement par le mot : OK",
        )
        text = response.text.strip()
        print(f"  OK Vertex AI opérationnel — modèle : {model}")
        print(f"     Réponse : {text}")
except Exception as e:
    msg = f"Erreur Vertex AI : {e}"
    print(f"  ERREUR {msg}")
    errors.append(msg)

# 6. Anonymisation des personnes (spaCy NER) — bloquant si le flag est actif
print("\n[6/6] Anonymisation des personnes (spaCy NER)...")
enable_person_anon = env_vals.get("ENABLE_PERSON_ANONYMIZATION", "true").strip().lower() not in ("false", "0", "no")
person_ner_model = env_vals.get("PERSON_NER_MODEL", "").strip() or "fr_core_news_sm"

if not enable_person_anon:
    print("  WARN ENABLE_PERSON_ANONYMIZATION=false — masquage des noms désactivé (filet de sécurité inactif)")
else:
    try:
        import spacy
        spacy.load(person_ner_model)
        print(f"  OK spaCy + modèle '{person_ner_model}' chargés")
    except Exception as e:
        msg = (
            f"ENABLE_PERSON_ANONYMIZATION=true mais le modèle spaCy '{person_ner_model}' "
            f"est introuvable/non chargeable : {e}"
        )
        print(f"  ERREUR {msg}")
        errors.append(msg)

# Résumé final
print("\n" + "=" * 55)
if not errors:
    print("  Environnement prêt — tu peux commencer !")
else:
    print(f"  {len(errors)} problème(s) à corriger :")
    for i, err in enumerate(errors, 1):
        print(f"     {i}. {err}")
print("=" * 55)
