"""
tests/generate_keyword_expansion.py
Génère automatiquement des mots-clés d'expansion de requête (via Gemini) pour
un sous-ensemble de questions du golden dataset, en n'exposant au modèle QUE
le champ question_client (jamais id_reference/categorie/reponse_attendue) —
garde-fou anti-fuite pour que le test reste une mesure d'expansion "réaliste"
(sans accès à la réponse).

Remplace la saisie manuelle (copier-coller AI Studio) responsable des 3
fichiers invalides précédents (contenu retapé/halluciné, désaligné du golden
dataset réel) par un pipeline reproductible : lecture stricte du golden
dataset → appel Gemini → validation stricte de la réponse → écriture du
rapport final.

Usage :
    python tests/generate_keyword_expansion.py
    python tests/generate_keyword_expansion.py --extra-ids gd_facturation_001,gd_facturation_014
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.config import get_settings

GOLDEN_DATASET_PATH = "tests/golden_dataset.json"
PROMPT_TEMPLATE_PATH = "ai_engine/prompts/query_expansion_keywords.txt"
OUTPUT_PATH = "tests/keyword_expansion_generated_v1.json"
PLACEHOLDER = "<INSERER_ICI_LE_JSON_DES_QUESTIONS>"

DEFAULT_ID_TESTS = ["gd_facturation_006", "gd_facturation_011"]


# ── Étape a-c : chargement strict du golden dataset ─────────────────────────

def load_golden_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def extract_target_questions(
    golden_questions: list[dict[str, Any]], id_tests_liste_blanche: list[str]
) -> list[dict[str, str]]:
    """
    Extrait UNIQUEMENT {id_test, question_client} pour chaque id_test de la
    liste blanche. Échoue explicitement si un id_test demandé n'existe pas
    dans le golden dataset (pas de continuation silencieuse).
    """
    golden_by_id = {q["id_test"]: q for q in golden_questions}

    manquants = [t for t in id_tests_liste_blanche if t not in golden_by_id]
    if manquants:
        raise ValueError(
            f"id_test demandé(s) absent(s) de {GOLDEN_DATASET_PATH} : {manquants}"
        )

    return [
        {"id_test": t, "question_client": golden_by_id[t]["question_client"]}
        for t in id_tests_liste_blanche
    ]


# ── Étape d : construction du prompt ────────────────────────────────────────

def load_prompt_template(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"Prompt introuvable : {path}")
    return Path(path).read_text(encoding="utf-8")


def build_prompt(template: str, questions: list[dict[str, str]]) -> str:
    if PLACEHOLDER not in template:
        raise ValueError(f"Placeholder {PLACEHOLDER!r} absent du template de prompt")
    questions_json = json.dumps(questions, ensure_ascii=False, indent=2)
    return template.replace(PLACEHOLDER, questions_json)


# ── Étape e : appel Gemini (pattern réutilisé de data_pipeline/extract_and_filter.py) ──

def build_gemini_client():
    from google import genai
    from google.oauth2 import service_account

    cfg = get_settings()
    credentials = service_account.Credentials.from_service_account_file(
        cfg.google_application_credentials,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return genai.Client(
        vertexai=True,
        project=cfg.gemini_project,
        location=cfg.gemini_location,
        credentials=credentials,
    ), cfg.gemini_model


def call_gemini(client, model: str, prompt: str) -> str:
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text.strip()


# ── Étape f : parsing + validation stricte ──────────────────────────────────

def parse_llm_json_response(raw_text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Aucun tableau JSON trouvé dans la réponse du modèle : {raw_text[:300]!r}")
    parsed = json.loads(match.group())
    if not isinstance(parsed, list):
        raise ValueError(f"Réponse JSON parsée n'est pas une liste : {type(parsed).__name__}")
    return parsed


def validate_llm_results(
    parsed_results: list[dict[str, Any]], id_tests_liste_blanche: list[str]
) -> list[dict[str, Any]]:
    """
    Le sous-ensemble d'id_test retourné par le modèle doit être EXACTEMENT
    celui envoyé — ni ajout, ni oubli. Échoue explicitement sinon.
    """
    attendus = set(id_tests_liste_blanche)
    recus = set()
    resultats_valides: list[dict[str, Any]] = []

    for entry in parsed_results:
        id_test = entry.get("id_test")
        mots_cles = entry.get("mots_cles_generes")

        if id_test is None:
            raise ValueError(f"Entrée sans champ 'id_test' dans la réponse du modèle : {entry}")
        if mots_cles is None or not isinstance(mots_cles, list):
            raise ValueError(f"Entrée {id_test!r} sans 'mots_cles_generes' valide (liste attendue) : {entry}")

        recus.add(id_test)
        resultats_valides.append({"id_test": id_test, "mots_cles_generes": mots_cles})

    invente = recus - attendus
    oublie = attendus - recus
    if invente or oublie:
        raise ValueError(
            "Écart entre id_test demandés et id_test retournés par le modèle : "
            f"inventé(s) (non demandé(s))={sorted(invente)}, oublié(s)={sorted(oublie)}"
        )

    return resultats_valides


# ── Étape g : sauvegarde ─────────────────────────────────────────────────────

def save_report(resultats: list[dict[str, Any]], model: str, output_path: str) -> None:
    report = {
        "generated_by": f"{model} (script automatisé)",
        "prompt_source": PROMPT_TEMPLATE_PATH,
        "date_generation": datetime.now(timezone.utc).isoformat(),
        "results": resultats,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(id_tests_liste_blanche: list[str]) -> dict[str, Any]:
    print(f"[INFO] Liste blanche d'id_test : {id_tests_liste_blanche}")

    golden_questions = load_golden_dataset(GOLDEN_DATASET_PATH)
    questions = extract_target_questions(golden_questions, id_tests_liste_blanche)
    print(f"[INFO] {len(questions)} question(s) extraite(s) (id_test + question_client uniquement)")
    for q in questions:
        print(f"  - {q['id_test']} : {q['question_client'][:80]}...")

    template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    prompt = build_prompt(template, questions)

    print("[INFO] Connexion Vertex AI / Gemini...")
    client, model = build_gemini_client()

    print(f"[INFO] Appel Gemini ({model})...")
    raw_response = call_gemini(client, model, prompt)
    print(f"[INFO] Réponse brute reçue ({len(raw_response)} caractères)")

    parsed = parse_llm_json_response(raw_response)
    resultats_valides = validate_llm_results(parsed, id_tests_liste_blanche)
    print(f"[OK] {len(resultats_valides)} résultat(s) validé(s) (sous-ensemble exact de la liste blanche)")

    save_report(resultats_valides, model, OUTPUT_PATH)
    print(f"[OK] Rapport sauvegardé dans {OUTPUT_PATH}")

    return {"results": resultats_valides}


def main():
    parser = argparse.ArgumentParser(
        description="Génère des mots-clés d'expansion de requête (Gemini) pour un sous-ensemble du golden dataset"
    )
    parser.add_argument(
        "--extra-ids", type=str, default=None,
        help="id_test additionnels séparés par des virgules (ex. Groupe A de contrôle), en plus de gd_facturation_006/011",
    )
    args = parser.parse_args()

    id_tests = list(DEFAULT_ID_TESTS)
    if args.extra_ids:
        id_tests.extend([t.strip() for t in args.extra_ids.split(",") if t.strip()])

    try:
        result = run(id_tests)
    except Exception as exc:
        print(f"\n[ERREUR FATALE] {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 80)
    print("CONTENU GÉNÉRÉ — VÉRIFICATION MANUELLE AVANT UTILISATION")
    print("=" * 80)
    for r in result["results"]:
        print(f"\n  {r['id_test']}")
        print(f"    mots_cles_generes : {r['mots_cles_generes']}")
    print()


if __name__ == "__main__":
    main()
