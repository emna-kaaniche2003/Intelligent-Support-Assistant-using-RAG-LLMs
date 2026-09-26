"""
Script d'inspection : récupère les 2 points Qdrant cibles et affiche côte à côte
- champ mots_cles du payload
- texte_pour_embedding réellement stocké/indexé (non recalculé)

Pour gd_facturation_006 (Devis, id_reference=280) et
gd_facturation_011 (Forfaits, id_reference=314).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient
from ai_engine.config import get_settings
from data_pipeline.ingestion.ingest_qr_technique import build_point_id

COLLECTION_NAME = "connaissances_modules"
SOURCE_TYPE = "qr_technique"

TARGETS = [
    {"id_reference": 280, "id_test": "gd_facturation_006", "categorie": "Devis"},
    {"id_reference": 314, "id_test": "gd_facturation_011", "categorie": "Forfaits"},
]


def main():
    print("=" * 80)
    print("INSPECTION DES POINTS QDRANT — mots_cles vs texte_pour_embedding")
    print("=" * 80)

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    for target in TARGETS:
        point_id = build_point_id(target["id_reference"], SOURCE_TYPE)
        print(f"\n--- {target['id_test']} | {target['categorie']} (id_ref={target['id_reference']}) ---")
        print(f"    Point UUID Qdrant : {point_id}")

        try:
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                print(f"    [ERREUR] Point non trouvé dans Qdrant")
                continue

            point = points[0]
            payload = point.payload

            print(f"\n    Champ mots_cles du payload :")
            mots_cles = payload.get("mots_cles")
            if mots_cles:
                print(f"      {mots_cles}")
            else:
                print(f"      [ABSENT ou vide]")

            print(f"\n    Champ question du payload :")
            print(f"      {payload.get('question', '[ABSENT]')[:120]}...")

            print(f"\n    Champ reponse du payload (tronqué) :")
            print(f"      {payload.get('reponse', '[ABSENT]')[:120]}...")

            print(f"\n    Champs métadata supplémentaires :")
            for k, v in sorted(payload.items()):
                if k not in ["mots_cles", "question", "reponse"]:
                    print(f"      {k} = {v}")

        except Exception as exc:
            print(f"    [ERREUR] Impossible de récupérer le point : {exc}")


if __name__ == "__main__":
    main()
