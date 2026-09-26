"""
test_motscles_ingest_variant.py
Ingère les 775 points qr_technique dans une collection Qdrant DE TEST
("connaissances_modules_test_motscles") avec une formule texte_pour_embedding
variante qui inclut mots_cles, pour mesurer l'effet sur le recall SANS
toucher à la collection de production "connaissances_modules".

Formule variante :
    "<categorie>. <mots_cles joints par ', '>. <question>. <question>. <reponse tronquée>"
(même troncature réponse=400 que l'original, pour isoler l'effet de l'ajout
des mots-clés seul — une seule variable change à la fois).

NE MODIFIE PAS data_pipeline/ingestion/ingest_qr_technique.py ni
data_pipeline/transform/remap_qr_technique.py : réutilise leurs briques
génériques (load_config, load_source, validate_entry, build_payload,
build_point_id, encode_dense_batch, build_bm25_index, vectorize_bm25,
iter_batches) sans dupliquer la formule elle-même, qui est réécrite ici
volontairement (c'est l'objet du test).

Usage :
    python test_motscles_ingest_variant.py
    python test_motscles_ingest_variant.py --config data_pipeline/config/ingestion_config_test_motscles.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ai_engine.config import get_settings
from data_pipeline.ingestion.ingest_qr_technique import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    build_bm25_index,
    build_payload,
    build_point_id,
    encode_dense_batch,
    iter_batches,
    load_config,
    load_source,
    validate_entry,
    vectorize_bm25,
)
from sentence_transformers import SentenceTransformer

DEFAULT_CONFIG_PATH = "data_pipeline/config/ingestion_config_test_motscles.json"


def build_texte_embedding_avec_mots_cles(entry: dict[str, Any], troncature: int) -> str:
    """
    Variante de build_texte_embedding (ingest_qr_technique.py:85-90) : insère
    mots_cles (joints par ", ") entre categorie et question. Seule variable
    modifiée par rapport à l'original — troncature réponse inchangée (400).
    """
    reponse = entry["reponse"]
    reponse_tronquee = reponse[:troncature] if len(reponse) > troncature else reponse
    question = entry["question"]
    mots_cles_str = ", ".join(entry["mots_cles"])
    return f"{entry['categorie']}. {mots_cles_str}. {question}. {question}. {reponse_tronquee}"


def prepare_entries_avec_mots_cles(
    raw_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[tuple[Any, str]]]:
    """Variante de prepare_entries (ingest_qr_technique.py) utilisant la formule avec mots_cles."""
    valid_entries: list[dict[str, Any]] = []
    texts: list[str] = []
    errors: list[tuple[Any, str]] = []

    for entry in raw_entries:
        entry_id = entry.get("id", "?")
        try:
            validate_entry(entry)
            texte_embedding = build_texte_embedding_avec_mots_cles(entry, config["longueur_troncature_reponse"])
            point_id = build_point_id(entry["id"], config["source_type"])
            payload = build_payload(entry, config)

            valid_entries.append({
                "point_id": point_id,
                "payload": payload,
                "source_id": entry["id"],
            })
            texts.append(texte_embedding)
        except Exception as exc:
            errors.append((entry_id, str(exc)))

    return valid_entries, texts, errors


def run(config_path: str) -> None:
    config = load_config(config_path)

    print(f"[INFO] Lecture source : {config['chemin_source']}")
    raw_entries = load_source(config["chemin_source"])
    print(f"[INFO] {len(raw_entries)} entrées source lues")

    valid_entries, texts, errors = prepare_entries_avec_mots_cles(raw_entries, config)

    for entry_id, message in errors:
        print(f"[ERREUR] Entrée id={entry_id} ignorée : {message}", file=sys.stderr)

    if not valid_entries:
        print("[INFO] Aucune entrée valide à ingérer.")
        return

    print(f"[INFO] Exemple de texte_pour_embedding (variante mots_cles) :")
    print(f"       {texts[0][:250]}...")

    print(f"[INFO] Chargement du modèle : {config['modele_embedding']}")
    model = SentenceTransformer(config["modele_embedding"])

    print("[INFO] Calcul des vecteurs sparse BM25 (corpus complet, formule avec mots_cles)…")
    bm25, vocab = build_bm25_index(texts)
    corpus_tokens = [t.lower().split() for t in texts]
    sparse_vecs = [vectorize_bm25(tokens, bm25, vocab) for tokens in corpus_tokens]

    cfg_settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {cfg_settings.qdrant_url}")
    client = QdrantClient(url=cfg_settings.qdrant_url, timeout=30)

    collection_name = config["collection_name"]
    if collection_name == "connaissances_modules":
        raise ValueError(
            "Garde-fou : ce script ne doit JAMAIS ingérer dans la collection de "
            "production 'connaissances_modules'. Vérifiez la config utilisée."
        )

    existing_count = client.count(collection_name).count
    if existing_count > 0:
        print(
            f"[INFO] La collection '{collection_name}' contient déjà {existing_count} points. "
            f"Ils seront écrasés (même point_id déterministe) par cet upsert."
        )

    batch_size = config["taille_batch"]
    indices = list(range(len(valid_entries)))
    batches = list(iter_batches(indices, batch_size))

    inserted = 0
    upsert_errors: list[tuple[Any, str]] = []

    print(f"[INFO] Upsert de {len(valid_entries)} points en {len(batches)} batch(es) dans '{collection_name}'…")
    for batch_indices in batches:
        batch_texts = [texts[i] for i in batch_indices]
        batch_dense = encode_dense_batch(model, batch_texts)

        points: list[PointStruct] = []
        for local_i, global_i in enumerate(batch_indices):
            entry = valid_entries[global_i]
            try:
                points.append(
                    PointStruct(
                        id=entry["point_id"],
                        vector={
                            DENSE_VECTOR_NAME: batch_dense[local_i],
                            SPARSE_VECTOR_NAME: sparse_vecs[global_i],
                        },
                        payload=entry["payload"],
                    )
                )
            except Exception as exc:
                upsert_errors.append((entry["source_id"], str(exc)))

        if points:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            inserted += len(points)

    for entry_id, message in upsert_errors:
        print(f"[ERREUR] Entrée id={entry_id} échec upsert : {message}", file=sys.stderr)

    total_errors = len(errors) + len(upsert_errors)
    print(f"\n[RESUME]")
    print(f"  Collection (TEST)       : {collection_name}")
    print(f"  Entrées source lues     : {len(raw_entries)}")
    print(f"  Nombre de batches       : {len(batches)}")
    print(f"  Points upsertés         : {inserted}")
    print(f"  Entrées en erreur       : {total_errors}")
    print(f"  [OK] Ingestion terminée (collection de test uniquement).\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion qr_technique variante mots_cles dans une collection Qdrant DE TEST"
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    try:
        run(args.config)
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
