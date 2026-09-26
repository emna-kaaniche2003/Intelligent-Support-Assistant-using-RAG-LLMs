"""
data_pipeline/ingest_qdrant.py
Ingestion des tickets résolus dans Qdrant avec hybrid search (dense + BM25 sparse).

Usage :
    python data_pipeline/ingest_qdrant.py --input data_pipeline/clean/reclamations_et_reponses.csv
    python data_pipeline/ingest_qdrant.py --input data_pipeline/clean/reclamations_et_reponses.xlsx --batch 64 --recreate
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Projet ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_engine.config import get_settings

# ── Constantes ────────────────────────────────────────────────────────────────
DENSE_DIM = 768          # paraphrase-multilingual-mpnet-base-v2
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Colonnes obligatoires dans le CSV/Excel d'entrée
REQUIRED_COLUMNS = {"titre", "description", "resolution"}

# Colonnes optionnelles conservées en payload
OPTIONAL_COLUMNS = ["categorie", "priorite", "statut", "date_creation", "ticket_id"]


# ── Chargement des données ─────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    """Charge un CSV ou Excel et valide les colonnes requises."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Format non supporté : {suffix}. Utiliser .csv ou .xlsx")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )

    # Supprimer les lignes sans résolution (inutilisables pour le RAG)
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    dropped = before - len(df)
    if dropped:
        print(f"[warn] {dropped} ligne(s) supprimées (champs obligatoires vides)")

    df = df.reset_index(drop=True)
    print(f"[info] {len(df)} tickets chargés depuis {path.name}")
    return df


def build_text(row: pd.Series) -> str:
    """Concatène titre + description + résolution pour l'embedding."""
    parts = [
        str(row.get("titre", "") or ""),
        str(row.get("description", "") or ""),
        str(row.get("resolution", "") or ""),
    ]
    return " ".join(p.strip() for p in parts if p.strip())


# ── Embeddings dense ───────────────────────────────────────────────────────────

def encode_dense(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode tous les textes en vecteurs denses (normalisation L2 incluse)."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ── Embeddings sparse BM25 ────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_sparse_vectors(texts: list[str]) -> list[SparseVector]:
    """
    Calcule les vecteurs BM25 sparse pour chaque texte.
    BM25Okapi entraîné sur l'ensemble du corpus → TF-IDF style scoring.
    """
    corpus_tokens = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    # Vocabulaire : mapping token → index
    vocab: dict[str, int] = {
        token: idx for idx, token in enumerate(bm25.idf.keys())
    }

    sparse_vecs: list[SparseVector] = []
    for tokens in tqdm(corpus_tokens, desc="BM25 sparse", unit="doc"):
        token_set = set(tokens)
        indices, values = [], []
        for token in token_set:
            if token in vocab:
                idx = vocab[token]
                # Score BM25 pour ce token dans ce document
                score = float(bm25.idf.get(token, 0.0))
                if score > 0:
                    indices.append(idx)
                    values.append(score)
        sparse_vecs.append(SparseVector(indices=indices, values=values))

    return sparse_vecs


# ── Collection Qdrant ─────────────────────────────────────────────────────────

def ensure_collection(client: QdrantClient, collection: str, recreate: bool) -> None:
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        print(f"[info] Collection '{collection}' supprimée (--recreate)")
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR: VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                    on_disk=False,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        print(f"[info] Collection '{collection}' créée (dense={DENSE_DIM}d + sparse BM25)")
    else:
        count = client.count(collection).count
        print(f"[info] Collection '{collection}' existante ({count} points)")


# ── Construction des points ───────────────────────────────────────────────────

def build_payload(row: pd.Series) -> dict:
    payload: dict = {
        "titre": str(row.get("titre", "") or ""),
        "description": str(row.get("description", "") or ""),
        "resolution": str(row.get("resolution", "") or ""),
    }
    for col in OPTIONAL_COLUMNS:
        val = row.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            payload[col] = str(val)
    return payload


def iter_batches(lst: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── Pipeline principal ────────────────────────────────────────────────────────

def ingest(
    input_path: Path,
    batch_size: int,
    recreate: bool,
) -> None:
    cfg = get_settings()

    # 1. Données
    df = load_data(input_path)
    texts = [build_text(row) for _, row in df.iterrows()]

    # 2. Modèle d'embedding
    print(f"[info] Chargement du modèle : {cfg.embedding_model}")
    model = SentenceTransformer(cfg.embedding_model)

    # 3. Vecteurs denses
    print("[info] Calcul des embeddings dense…")
    dense_vecs = encode_dense(model, texts, batch_size)

    # 4. Vecteurs sparse BM25
    print("[info] Calcul des vecteurs sparse BM25…")
    sparse_vecs = build_sparse_vectors(texts)

    # 5. Connexion Qdrant
    print(f"[info] Connexion à Qdrant : {cfg.qdrant_url}")
    client = QdrantClient(url=cfg.qdrant_url, timeout=30)
    ensure_collection(client, cfg.qdrant_collection, recreate)

    # 6. Upsert par batches
    n = len(df)
    all_points: list[PointStruct] = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Utiliser ticket_id existant ou générer un UUID déterministe
        raw_id = row.get("ticket_id")
        if raw_id is not None and not (isinstance(raw_id, float) and np.isnan(raw_id)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))
        else:
            point_id = str(uuid.uuid4())

        all_points.append(
            PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR: dense_vecs[i].tolist(),
                    SPARSE_VECTOR: sparse_vecs[i],
                },
                payload=build_payload(row),
            )
        )

    print(f"[info] Upsert de {n} points (batch={batch_size})…")
    inserted = 0
    for batch in tqdm(list(iter_batches(all_points, batch_size)), desc="Upsert", unit="batch"):
        client.upsert(collection_name=cfg.qdrant_collection, points=batch, wait=True)
        inserted += len(batch)

    # 7. Vérification finale
    final_count = client.count(cfg.qdrant_collection).count
    print(f"\n[ok] Ingestion terminée — {final_count} points dans '{cfg.qdrant_collection}'")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingestion de tickets résolus dans Qdrant (hybrid search dense + BM25)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Chemin vers le fichier de tickets (CSV ou XLSX)",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=32,
        help="Taille des batches d'upsert (défaut : 32)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Supprimer et recréer la collection avant l'ingestion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.input.exists():
        print(f"[erreur] Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)
    ingest(args.input, args.batch, args.recreate)
