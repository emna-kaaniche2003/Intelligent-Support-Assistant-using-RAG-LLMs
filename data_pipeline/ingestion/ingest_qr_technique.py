"""
data_pipeline/ingestion/ingest_qr_technique.py
Ingestion des questions/réponses qr_technique (facturation) dans Qdrant,
collection "connaissances_modules" (hybrid search dense + BM25 sparse).

Usage :
    python data_pipeline/ingestion/ingest_qr_technique.py
    python data_pipeline/ingestion/ingest_qr_technique.py --config data_pipeline/config/ingestion_config.json

Pattern d'ingestion hybride réutilisé depuis data_pipeline/ingest_qdrant.py
(pipeline tickets_resolus) — mêmes briques : SentenceTransformer pour le dense,
rank_bm25.BM25Okapi (IDF par token, pas le score BM25 complet) pour le sparse,
PointStruct avec vecteurs nommés, upsert par lots. Écarts volontaires documentés
ci-dessous :

1. Noms de vecteurs "dense"/"bm25" (au lieu de "sparse" dans le script original) :
   imposés par le schéma déjà créé de la collection "connaissances_modules"
   (cf. data_pipeline/ingestion/create_collection.py + qdrant_config.json).
2. ID de point = uuid5 dérivé de "<source_type>_<id d'origine>" (pas de l'id brut
   comme dans le script original) : la collection "connaissances_modules" est
   destinée à accueillir plusieurs sources (qr_technique, doc_module, plus tard
   reclamations) — préfixer par source_type évite toute collision d'UUID entre
   deux sources qui partageraient le même entier d'id d'origine.
3. Encodage dense appelé explicitement une fois par batch (plutôt qu'un seul appel
   model.encode(texts, batch_size=N) sur tout le corpus comme dans l'original) :
   permet d'isoler l'upsert par lot (log par batch) et de vérifier en test le
   nombre d'appels réel à l'encodeur. Le résultat numérique est équivalent (même
   modèle, même paramètres de normalisation), seul le découpage des appels change.
4. Le score BM25 sparse est calculé une seule fois sur l'ensemble du corpus valide
   (comme l'original) : l'IDF est une statistique globale, non batchable.
5. Validation champ-par-champ par entrée (try/except), absente du script original
   qui ne valide que les colonnes d'un DataFrame (la source JSON ici est une liste
   d'objets, sans garde-fou de colonnes équivalent) — une entrée malformée est
   journalisée comme erreur et n'interrompt pas le traitement des autres.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Generator

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_engine.config import get_settings

DEFAULT_CONFIG_PATH = "data_pipeline/config/ingestion_config.json"

# Noms de vecteurs imposés par le schéma de la collection "connaissances_modules"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

REQUIRED_FIELDS = ["id", "categorie", "source", "ecran", "question", "reponse", "mots_cles"]


# ── Chargement ────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source(chemin_source: str) -> list[dict[str, Any]]:
    with open(chemin_source, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("questions", [])


# ── Validation et transformation d'une entrée ────────────────────────────────

def validate_entry(entry: dict[str, Any]) -> None:
    """Lève une exception explicite si un champ requis est manquant."""
    for field in REQUIRED_FIELDS:
        if field not in entry:
            raise ValueError(f"champ requis '{field}' manquant")


def build_texte_embedding(entry: dict[str, Any], troncature: int) -> str:
    """categorie. question. question. reponse tronquée."""
    reponse = entry["reponse"]
    reponse_tronquee = reponse[:troncature] if len(reponse) > troncature else reponse
    question = entry["question"]
    return f"{entry['categorie']}. {question}. {question}. {reponse_tronquee}"


def build_payload(entry: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": config["module"],
        "source_type": config["source_type"],
        "categorie": entry["categorie"],
        "ecran": entry["ecran"],
        "interface_ou_serveur": entry["source"],
        "question": entry["question"],
        "reponse": entry["reponse"],
        "mots_cles": entry["mots_cles"],
    }


def build_point_id(entry_id: Any, source_type: str) -> str:
    """UUID déterministe namespacé par source_type (évite collision inter-sources)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_type}_{entry_id}"))


def prepare_entries(
    raw_entries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[tuple[Any, str]]]:
    """
    Valide et transforme chaque entrée brute.
    Retourne (entrées_valides_enrichies, textes_pour_embedding, erreurs[(id, message)]).
    Une entrée malformée est journalisée dans errors et exclue, sans interrompre le reste.
    """
    valid_entries: list[dict[str, Any]] = []
    texts: list[str] = []
    errors: list[tuple[Any, str]] = []

    for entry in raw_entries:
        entry_id = entry.get("id", "?")
        try:
            validate_entry(entry)
            texte_embedding = build_texte_embedding(entry, config["longueur_troncature_reponse"])
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


# ── Embeddings dense ──────────────────────────────────────────────────────────

def encode_dense_batch(model: SentenceTransformer, batch_texts: list[str]) -> list[list[float]]:
    """Encode un seul batch de textes (un appel = un batch, cf. écart #3 documenté en tête de module)."""
    vecs = model.encode(
        batch_texts,
        batch_size=len(batch_texts),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


# ── Embeddings sparse BM25 (identique au pattern ingest_qdrant.py) ───────────

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_bm25_index(texts: list[str]) -> tuple[BM25Okapi, dict[str, int]]:
    """
    Entraîne BM25Okapi sur le corpus et construit le vocabulaire token → index.
    Exposé séparément (plutôt que noyé dans build_sparse_vectors) pour permettre
    à ai_engine/rag/evaluate_retrieval.py de vectoriser une question hors-corpus
    (golden dataset) avec exactement le même vocabulaire/IDF que l'ingestion.
    """
    corpus_tokens = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)
    vocab: dict[str, int] = {token: idx for idx, token in enumerate(bm25.idf.keys())}
    return bm25, vocab


def vectorize_bm25(tokens: list[str], bm25: BM25Okapi, vocab: dict[str, int]) -> SparseVector:
    """Vecteur sparse (IDF par token) pour une liste de tokens donnée, selon un index BM25 déjà construit."""
    token_set = set(tokens)
    indices, values = [], []
    for token in token_set:
        if token in vocab:
            score = float(bm25.idf.get(token, 0.0))
            if score > 0:
                indices.append(vocab[token])
                values.append(score)
    return SparseVector(indices=indices, values=values)


def build_sparse_vectors(texts: list[str]) -> list[SparseVector]:
    """BM25Okapi entraîné sur tout le corpus valide → IDF par token (poids sparse)."""
    corpus_tokens = [tokenize(t) for t in texts]
    bm25, vocab = build_bm25_index(texts)
    return [vectorize_bm25(tokens, bm25, vocab) for tokens in corpus_tokens]


# ── Batching générique ────────────────────────────────────────────────────────

def iter_batches(lst: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    config = load_config(config_path)

    print(f"[INFO] Lecture source : {config['chemin_source']}")
    raw_entries = load_source(config["chemin_source"])
    print(f"[INFO] {len(raw_entries)} entrées source lues")

    valid_entries, texts, errors = prepare_entries(raw_entries, config)

    for entry_id, message in errors:
        print(f"[ERREUR] Entrée id={entry_id} ignorée : {message}", file=sys.stderr)

    if not valid_entries:
        print("[INFO] Aucune entrée valide à ingérer.")
        return

    print(f"[INFO] Chargement du modèle : {config['modele_embedding']}")
    model = SentenceTransformer(config["modele_embedding"])

    print("[INFO] Calcul des vecteurs sparse BM25 (corpus complet)…")
    sparse_vecs = build_sparse_vectors(texts)

    cfg_settings = get_settings()
    print(f"[INFO] Connexion à Qdrant : {cfg_settings.qdrant_url}")
    client = QdrantClient(url=cfg_settings.qdrant_url, timeout=30)

    batch_size = config["taille_batch"]
    indices = list(range(len(valid_entries)))
    batches = list(iter_batches(indices, batch_size))

    inserted = 0
    upsert_errors: list[tuple[Any, str]] = []

    print(f"[INFO] Upsert de {len(valid_entries)} points en {len(batches)} batch(es) (taille={batch_size})…")
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
            client.upsert(collection_name=config["collection_name"], points=points, wait=True)
            inserted += len(points)

    for entry_id, message in upsert_errors:
        print(f"[ERREUR] Entrée id={entry_id} échec upsert : {message}", file=sys.stderr)

    total_errors = len(errors) + len(upsert_errors)
    print(f"\n[RESUME]")
    print(f"  Entrées source lues     : {len(raw_entries)}")
    print(f"  Nombre de batches       : {len(batches)}")
    print(f"  Points upsertés         : {inserted}")
    print(f"  Entrées en erreur       : {total_errors}")
    print(f"  [OK] Ingestion terminée.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion qr_technique (facturation) dans Qdrant (connaissances_modules)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Chemin du fichier config JSON (défaut : {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    try:
        run(args.config)
    except Exception as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
