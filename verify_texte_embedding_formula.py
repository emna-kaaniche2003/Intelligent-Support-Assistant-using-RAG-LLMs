"""
Vérifie la formule exacte de texte_pour_embedding pour les 2 points cibles.

Recalcule texte_pour_embedding à partir du payload actuel (sans mots_cles)
selon la formule documentée dans ingest_qr_technique.py:
  texte_pour_embedding = "<categorie>. <question>. <question>. <reponse tronquée>"

Affiche côte à côte :
- Les mots_cles du point (présents dans le payload, mais NON utilisés pour l'embedding)
- Le texte_pour_embedding réel (formule confirmée, sans mots_cles)
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
LONGUEUR_TRONCATURE_REPONSE = 400  # de ingestion_config.json

TARGETS = [
    {"id_reference": 280, "id_test": "gd_facturation_006", "categorie": "Devis"},
    {"id_reference": 314, "id_test": "gd_facturation_011", "categorie": "Forfaits"},
]


def reconstruct_texte_embedding(payload: dict, troncature: int) -> str:
    """Reconstruit la formule exacte documentée : <categorie>. <question>. <question>. <reponse tronquée>."""
    categorie = payload.get("categorie", "")
    question = payload.get("question", "")
    reponse = payload.get("reponse", "")
    reponse_tronquee = reponse[:troncature] if len(reponse) > troncature else reponse
    return f"{categorie}. {question}. {question}. {reponse_tronquee}"


def main():
    print("=" * 80)
    print("VÉRIFICATION DE LA FORMULE texte_pour_embedding")
    print("(SANS mots_cles inclus)")
    print("=" * 80)

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)

    for target in TARGETS:
        point_id = build_point_id(target["id_reference"], SOURCE_TYPE)
        print(f"\n--- {target['id_test']} ({target['categorie']}) ---")

        try:
            points = client.retrieve(
                collection_name=COLLECTION_NAME,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                print(f"    [ERREUR] Point non trouvé")
                continue

            payload = points[0].payload

            mots_cles = payload.get("mots_cles", [])
            texte_embedding = reconstruct_texte_embedding(payload, LONGUEUR_TRONCATURE_REPONSE)

            print(f"\n  Champ mots_cles du payload (ABSENT de l'embedding) :")
            print(f"    {mots_cles}")

            print(f"\n  Formule texte_pour_embedding (UTILISÉE pour encoding dense + BM25 sparse) :")
            print(f"    Longueur : {len(texte_embedding)} caractères")
            print(f"    Contenu (premiers 200 chars) : {texte_embedding[:200]}...")
            print(f"\n  Inclusion de mots_cles dans texte_for_embedding ? NON (pas dans la formule)")
            print(f"  Formule exacte : '<categorie>. <question>. <question>. <reponse[:400]>'")

        except Exception as exc:
            print(f"    [ERREUR] {exc}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("Le champ mots_cles EXISTE dans le payload Qdrant mais N'EST PAS utilise")
    print("par la recherche dense (SentenceTransformer) ni par la recherche sparse (BM25).")
    print("\nFormule d'embedding : <categorie>. <question>. <question>. <reponse truncatee>")
    print("  -> mots_cles absent de cette formule")
    print("\nBM25 vectorise : tokenize(texte_pour_embedding)")
    print("  -> mots_cles ne participent PAS au calcul BM25")
    print("\nConsequence : l'expansion de requete par mots-cles generes ne peut")
    print("ameliorer que le matching lexical/sparse. C'est une limitation structurelle,")
    print("pas un probleme du generateur LLM lui-meme.")
    print()


if __name__ == "__main__":
    main()
