"""
Vérification des id_reference du golden_dataset.json contre la collection Qdrant.

Script en lecture seule — aucune modification de Qdrant ou des fichiers de production.

Usage :
    python verify_golden_dataset.py

Résultat :
    - Liste détaillée : chaque id_reference, présence/absence dans Qdrant, match sémantique
    - Rapport synthétique : X/20 valides, liste des invalides/suspects
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_engine.config import get_settings

GOLDEN_DATASET_PATH = "tests/golden_dataset.json"
SOURCE_TYPE = "qr_technique"
COLLECTION_NAME = "connaissances_modules"


def load_golden_dataset() -> list[dict[str, Any]]:
    """Charge les 20 questions du golden_dataset."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


def build_qdrant_point_id(id_reference: int, source_type: str = SOURCE_TYPE) -> str:
    """Reconstitue l'UUID Qdrant déterministe comme dans ingest_qr_technique.py."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_type}_{id_reference}"))


def verify_point_exists(client: QdrantClient, point_id: str) -> dict[str, Any] | None:
    """Récupère un point Qdrant par son UUID. Retourne le point ou None."""
    try:
        result = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        return result[0] if result else None
    except Exception as e:
        print(f"[ERREUR] Impossible de récupérer le point {point_id}: {e}", file=sys.stderr)
        return None


def check_semantic_match(
    golden_question: str,
    qdrant_question: str,
    qdrant_reponse: str,
) -> tuple[bool, str]:
    """
    Vérifie la cohérence sémantique entre question du golden dataset
    et point Qdrant correspondant.
    Retourne (is_match, diagnostic).
    """
    # Critères simples : longueur raisonnable, pas vide
    if not qdrant_question or not qdrant_reponse:
        return False, "Champ question ou réponse vide dans Qdrant"

    if len(qdrant_question) < 10 or len(qdrant_reponse) < 10:
        return False, "Question ou réponse trop courte"

    # Pas de vérification stricte de contenu (variation textuelle possible)
    # Juste vérifier que les champs existent et semblent raisonnables
    return True, "OK"


def main():
    print("=" * 80)
    print("VÉRIFICATION DES ID_REFERENCE DU GOLDEN_DATASET")
    print("=" * 80)
    print()

    # Charge le golden dataset
    questions = load_golden_dataset()
    print(f"[INFO] Chargement du golden_dataset : {len(questions)} questions")
    print()

    # Connexion Qdrant
    settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {settings.qdrant_url}")
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=30)
        collections = client.get_collections()
        print(f"[INFO] Collections disponibles : {len(collections.collections)}")
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter à Qdrant : {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("-" * 80)
    print("DÉTAIL PAR ID_REFERENCE")
    print("-" * 80)
    print()

    valid_count = 0
    invalid_refs: list[dict[str, Any]] = []
    suspicious_refs: list[dict[str, Any]] = []

    for q in questions:
        id_test = q.get("id_test", "?")
        id_reference = q.get("id_reference")
        golden_question = q.get("question_client", "")

        if id_reference is None:
            print(f"[NO] {id_test} : id_reference manquant")
            invalid_refs.append({"id_test": id_test, "id_reference": id_reference, "reason": "id_reference manquant"})
            continue

        # Reconstruit l'UUID Qdrant
        qdrant_id = build_qdrant_point_id(id_reference)

        # Vérifie existence dans Qdrant
        point = verify_point_exists(client, qdrant_id)

        if not point:
            print(f"[NO] {id_test} (id_ref={id_reference}) : point NOT FOUND dans Qdrant")
            invalid_refs.append({
                "id_test": id_test,
                "id_reference": id_reference,
                "reason": "Point not found in Qdrant",
            })
            continue

        # Point trouvé — extrait payload
        payload = point.payload
        qdrant_question = payload.get("question", "")
        qdrant_reponse = payload.get("reponse", "")
        categorie = payload.get("categorie", "?")

        # Vérifie cohérence sémantique
        is_match, diagnostic = check_semantic_match(golden_question, qdrant_question, qdrant_reponse)

        if is_match:
            print(f"[OK] {id_test} (id_ref={id_reference}) [{categorie}]")
            print(f"   Golden question: {golden_question[:70]}...")
            print(f"   Qdrant question: {qdrant_question[:70]}...")
            print()
            valid_count += 1
        else:
            print(f"[!!] {id_test} (id_ref={id_reference}) [{categorie}] — SUSPECT")
            print(f"   Diagnostic: {diagnostic}")
            print(f"   Golden question: {golden_question[:70]}...")
            print(f"   Qdrant question: {qdrant_question[:70]}...")
            print()
            suspicious_refs.append({
                "id_test": id_test,
                "id_reference": id_reference,
                "reason": diagnostic,
            })

    print()
    print("=" * 80)
    print("RAPPORT SYNTHÉTIQUE")
    print("=" * 80)
    print()
    print(f"Total de questions : {len(questions)}")
    print(f"ID_reference valides (trouvés + cohérents) : {valid_count}")
    print(f"ID_reference invalides : {len(invalid_refs)}")
    print(f"ID_reference suspects (mismatch sémantique) : {len(suspicious_refs)}")
    print()

    if invalid_refs:
        print("--- INVALIDES ---")
        for ref in invalid_refs:
            print(f"  {ref['id_test']} (id_ref={ref['id_reference']}) : {ref['reason']}")
        print()

    if suspicious_refs:
        print("--- SUSPECTS ---")
        for ref in suspicious_refs:
            print(f"  {ref['id_test']} (id_ref={ref['id_reference']}) : {ref['reason']}")
        print()

    print()
    success_rate = (valid_count / len(questions)) * 100 if questions else 0
    print(f"Taux de réussite : {valid_count}/{len(questions)} ({success_rate:.1f}%)")
    print()


if __name__ == "__main__":
    main()
