"""
data_pipeline/ingestion/create_collection.py
Création de la collection Qdrant "connaissances_modules" (hybrid search dense + sparse BM25).

Usage :
    python data_pipeline/ingestion/create_collection.py
    python data_pipeline/ingestion/create_collection.py --config data_pipeline/config/qdrant_config.json
    python data_pipeline/ingestion/create_collection.py --force-recreate

Note API (qdrant-client==1.9.2, épinglé dans requirements.txt) :
    `models.Modifier.IDF` (poids IDF calculé côté serveur pour les sparse vectors)
    n'existe qu'à partir de qdrant-client 1.10+ / Qdrant server 1.10+ — absent de la
    1.9.2 installée ici (vérifié : SparseVectorParams n'a qu'un champ `index`, pas de
    `modifier`). Le sparse vector est donc déclaré sans modifier serveur ; le poids
    BM25 doit être calculé côté client avant upsert (ex. rank_bm25.BM25Okapi, déjà
    utilisé dans ingest_qdrant.py) plutôt que délégué au serveur.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

DEFAULT_CONFIG_PATH = "data_pipeline/config/qdrant_config.json"

_DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
    "Manhattan": Distance.MANHATTAN,
}

_PAYLOAD_TYPE_MAP = {
    "keyword": PayloadSchemaType.KEYWORD,
}


def load_config(config_path: str) -> dict[str, Any]:
    """Charge la config JSON de la collection Qdrant."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_distance(distance_name: str) -> Distance:
    if distance_name not in _DISTANCE_MAP:
        raise ValueError(
            f"Distance '{distance_name}' inconnue. Valeurs supportées : "
            f"{list(_DISTANCE_MAP.keys())}"
        )
    return _DISTANCE_MAP[distance_name]


def resolve_payload_schema(index_type: str) -> PayloadSchemaType:
    if index_type not in _PAYLOAD_TYPE_MAP:
        raise ValueError(
            f"Type d'index payload '{index_type}' non supporté. Valeurs supportées : "
            f"{list(_PAYLOAD_TYPE_MAP.keys())}"
        )
    return _PAYLOAD_TYPE_MAP[index_type]


def connect_client(host: str, port: int) -> QdrantClient:
    """Se connecte à Qdrant, avec message explicite si l'instance n'est pas joignable."""
    client = QdrantClient(host=host, port=port)
    try:
        client.get_collections()
    except Exception as exc:
        raise ConnectionError(
            f"Impossible de joindre Qdrant sur {host}:{port}. "
            f"Vérifiez que le conteneur Docker est bien démarré "
            f"(ex. `docker ps` doit lister un conteneur qdrant sur le port {port}).\n"
            f"Détail de l'erreur : {exc}"
        ) from exc
    return client


def collection_already_exists(client: QdrantClient, collection_name: str, force_recreate: bool) -> bool:
    """
    Retourne True si la collection existe et doit être conservée telle quelle
    (c'est-à-dire : existe ET force_recreate est False).
    Si force_recreate est True et que la collection existe, la supprime ici
    (avec confirmation affichée) puis retourne False pour signaler qu'il faut la recréer.
    """
    exists = client.collection_exists(collection_name)

    if not exists:
        return False

    if not force_recreate:
        count = client.count(collection_name).count
        print(
            f"[INFO] La collection '{collection_name}' existe déjà ({count} points). "
            f"Aucune action effectuée (utilisez --force-recreate pour la recréer)."
        )
        return True

    # force_recreate=True et la collection existe : confirmation puis suppression
    count = client.count(collection_name).count
    print(
        f"[ATTENTION] Suppression de la collection '{collection_name}' "
        f"({count} points actuellement stockés) suite à --force-recreate."
    )
    client.delete_collection(collection_name)
    print(f"[INFO] Collection '{collection_name}' supprimée.")
    return False


def create_collection(client: QdrantClient, config: dict[str, Any]) -> None:
    """Crée la collection avec vecteur dense + sparse BM25 selon la config."""
    collection_name = config["collection_name"]
    dense_name = config["dense_vector_name"]
    dense_size = config["dense_vector_size"]
    distance = resolve_distance(config["distance"])
    sparse_name = config["sparse_vector_name"]

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            dense_name: VectorParams(
                size=dense_size,
                distance=distance,
            ),
        },
        sparse_vectors_config={
            sparse_name: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )


def create_payload_indexes(client: QdrantClient, config: dict[str, Any]) -> list[str]:
    """Crée les index de payload listés dans la config. Retourne la liste des champs indexés."""
    created_fields = []
    for index_spec in config["payload_indexes"]:
        field_name = index_spec["field"]
        field_schema = resolve_payload_schema(index_spec["type"])
        client.create_payload_index(
            collection_name=config["collection_name"],
            field_name=field_name,
            field_schema=field_schema,
        )
        created_fields.append(field_name)
    return created_fields


def run(config_path: str, force_recreate: bool) -> None:
    config = load_config(config_path)

    client = connect_client(config["host"], config["port"])

    if collection_already_exists(client, config["collection_name"], force_recreate):
        return

    create_collection(client, config)
    created_fields = create_payload_indexes(client, config)

    print(f"\n[RESUME]")
    print(f"  Collection créée : {config['collection_name']}")
    print(
        f"  Vecteur dense    : nom='{config['dense_vector_name']}', "
        f"taille={config['dense_vector_size']}, distance={config['distance']}"
    )
    print(f"  Vecteur sparse   : nom='{config['sparse_vector_name']}' (sans modifier serveur, cf. docstring module)")
    print(f"  Index payload    : {created_fields}")
    print(f"  [OK] Collection prête pour ingestion.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Crée la collection Qdrant 'connaissances_modules' (dense + sparse BM25)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Chemin du fichier config JSON (défaut : {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Supprime puis recrée la collection si elle existe déjà",
    )
    args = parser.parse_args()

    try:
        run(args.config, args.force_recreate)
    except ConnectionError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
