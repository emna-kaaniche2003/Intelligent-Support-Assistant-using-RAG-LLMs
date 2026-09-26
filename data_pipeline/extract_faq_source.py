"""
data_pipeline/extract_faq_source.py
Extraction des réclamations RÉPÉTITIVES (contenu client uniquement, sans
résolution) pour le pipeline FAQ : rédaction de réponses types par l'équipe
support avant ingestion dans le RAG.

Pipeline :
  raw/messages_faq.csv (export via queries/extract_tickets_faq.sql :
  idMessage, objet, message, module_designation, objectif_reclamation, IdClient —
  déjà filtré par module/objectif en SQL, revalidé ici en Python. IdClient
  toujours conservé en sortie, rempli à vide + warning s'il est absent de la
  source)
    → nettoyage HTML + anonymisation + généricisation des numéros métier
      (cleaning_utils.py)
    → filtrage module (Financier/Facturation/Comptabilité)
      + objectif (Assistance/Formation) + longueur minimale
    → clean/reclamations_seules.csv + clean/reclamations_seules_filter_report.csv

Contrairement à extract_and_filter.py : pas de filtre LLM à cette étape
(rule-based uniquement). Le dédoublonnage sémantique se fera dans une étape
suivante (bucketing par mots-clés, cf. data_pipeline/config/module_keywords.json).

Usage :
    python data_pipeline/extract_faq_source.py \\
        --input data_pipeline/raw/messages_faq.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.cleaning_utils import (
    anonymize_text,
    clean_html,
    genericize_business_numbers,
    hash_text,
    is_noise,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Constantes ────────────────────────────────────────────────────────────────

REQUIRED_COLS = ["idMessage", "objet", "message", "module_designation", "objectif_reclamation"]

OUTPUT_DIR = Path(__file__).resolve().parent / "clean"

VALID_MODULES = {"financier", "facturation", "comptabilité"}
VALID_OBJECTIFS = {"assistance", "formation"}

MIN_TEXTE_LENGTH_DEFAULT = 25


# ── Chargement ────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    # Auto-détection séparateur : essaye ; puis ,
    df = None
    for sep in (";", ","):
        try:
            candidate = pd.read_csv(path, encoding="utf-8-sig", sep=sep, low_memory=False)
            if len(candidate.columns) > 1:
                df = candidate
                break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"Impossible de lire {path} avec les séparateurs ';' ou ','")

    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(
            f"[erreur] Colonnes obligatoires manquantes : {missing}\n"
            f"         Colonnes disponibles : {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    if "IdClient" not in df.columns:
        print("[warn] Colonne 'IdClient' absente du CSV d'entrée — colonne de sortie remplie à vide")
        df["IdClient"] = pd.NA

    print(f"[info] {len(df)} lignes chargées depuis {path.name}")
    print(f"[info] Colonnes détectées : {list(df.columns)}")
    return df[REQUIRED_COLS + ["IdClient"]].copy()


# ── Normalisation catégories ──────────────────────────────────────────────────

def normalize_category(value: object) -> str:
    """
    Normalise casse et espaces superflus pour le FILTRAGE (ex. 'Facturation  '
    -> 'facturation'). Ne sert qu'à comparer avec VALID_MODULES/VALID_OBJECTIFS,
    pas à la valeur stockée (cf. normalize_stored_category).
    """
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def normalize_stored_category(value: object) -> object:
    """
    Normalise les espaces d'une valeur catégorielle AVANT écriture dans le CSV
    (strip + runs d'espaces -> un espace), en conservant casse et accents pour
    l'affichage ('Comptabilité ' -> 'Comptabilité', 'Facturation  ' ->
    'Facturation'). Correctif à la source : l'export SQL laisse des espaces en
    fin qui cassaient le matching aval par égalité (bucketing de
    detect_reclamations_redondantes.py). Les non-str (NA) sont laissés tels quels.
    """
    if not isinstance(value, str):
        return value
    return re.sub(r"\s+", " ", value).strip()


def _display(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "(vide)"


# ── Nettoyage + filtrage ──────────────────────────────────────────────────────

def process(df: pd.DataFrame, min_length: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Nettoie le HTML + anonymise objet/message
    2. Applique les règles de filtrage (module, objectif, longueur)
    Retourne (reclamations_retenues, rapport_complet).
    """
    df = df.copy()
    if "IdClient" not in df.columns:
        df["IdClient"] = pd.NA

    # Correctif à la source : strip des valeurs catégorielles stockées (l'export
    # SQL laisse des espaces en fin type "Facturation  "/"Comptabilité " qui
    # cassaient le matching aval par égalité). Fait AVANT écriture, en plus du
    # strip déjà présent dans normalize_category() (lui limité au filtrage).
    for col in ("module_designation", "objectif_reclamation"):
        df[col] = df[col].apply(normalize_stored_category)

    print("[info] Nettoyage HTML + anonymisation en cours…")
    df["objet_clean"] = (
        df["objet"].apply(clean_html).apply(anonymize_text).apply(genericize_business_numbers)
    )
    df["message_clean"] = (
        df["message"].apply(clean_html).apply(anonymize_text).apply(genericize_business_numbers)
    )
    df["texte_complet"] = (df["objet_clean"] + " " + df["message_clean"]).str.strip()
    df["longueur_texte"] = df["texte_complet"].str.len()
    df["hash_doublon"] = df["texte_complet"].apply(hash_text)

    reject_mask = pd.Series(False, index=df.index)
    reasons: dict[int, str] = {}

    # 1. Module hors périmètre (Financier/Facturation/Comptabilité)
    module_norm = df["module_designation"].apply(normalize_category)
    mask = ~module_norm.isin(VALID_MODULES)
    for idx in df[mask & ~reject_mask].index:
        reasons[idx] = f"module hors périmètre ('{_display(df.loc[idx, 'module_designation'])}')"
    reject_mask |= mask

    # 2. Objectif hors périmètre (Assistance/Formation)
    objectif_norm = df["objectif_reclamation"].apply(normalize_category)
    mask = ~objectif_norm.isin(VALID_OBJECTIFS)
    for idx in df[mask & ~reject_mask].index:
        reasons[idx] = f"objectif hors périmètre ('{_display(df.loc[idx, 'objectif_reclamation'])}')"
    reject_mask |= mask

    # 3. Texte trop court après nettoyage
    mask = df["texte_complet"].apply(lambda t: is_noise(t, min_length))
    for idx in df[mask & ~reject_mask].index:
        reasons[idx] = f"texte trop court (< {min_length} chars)"
    reject_mask |= mask

    n_dup = int(df.loc[~reject_mask, "hash_doublon"].duplicated(keep="first").sum())

    kept_cols = [
        "idMessage", "module_designation", "objectif_reclamation", "IdClient",
        "objet_clean", "message_clean", "texte_complet", "longueur_texte", "hash_doublon",
    ]
    kept = df[~reject_mask][kept_cols].reset_index(drop=True)

    rejected = df[reject_mask].copy()
    rejected["statut_filtre"] = "rejeté"
    rejected["raison_rejet"] = rejected.index.map(reasons)

    report_kept = pd.DataFrame({
        "idMessage": kept["idMessage"],
        "statut_filtre": "retenu",
        "raison_rejet": "",
    })
    report_rejected = rejected[["idMessage", "statut_filtre", "raison_rejet"]].reset_index(drop=True)
    report = pd.concat([report_kept, report_rejected], ignore_index=True)

    print(
        f"[règles] {len(df)} → {len(kept)} retenus "
        f"({len(df) - len(kept)} rejetés)"
        f"{f' — {n_dup} doublons exacts détectés (non supprimés, marqués via hash_doublon)' if n_dup else ''}"
    )
    return kept, report


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_outputs(kept: pd.DataFrame, report: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = OUTPUT_DIR / "reclamations_seules.csv"
    kept.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"[ok] reclamations_seules.csv → {clean_path} ({len(kept)} réclamations)")

    report_path = OUTPUT_DIR / "reclamations_seules_filter_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[ok] reclamations_seules_filter_report.csv → {report_path} ({len(report)} lignes total)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraction des réclamations seules (sans résolution) pour le pipeline FAQ"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Chemin vers le CSV brut (export extract_tickets_faq.sql)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=MIN_TEXTE_LENGTH_DEFAULT,
        help=f"Longueur minimale du texte complet après nettoyage (défaut : {MIN_TEXTE_LENGTH_DEFAULT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"[erreur] Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    df = load_raw(args.input)
    kept, report = process(df, args.min_length)
    save_outputs(kept, report)

    print(f"\n[résumé] {len(df)} bruts → {len(kept)} retenus pour bucketing/FAQ")
    print(
        "[suivant] bucketing par mots-clés "
        "(data_pipeline/config/module_keywords.json) "
        "+ génération LLM de questions canoniques (à venir)"
    )


if __name__ == "__main__":
    main()
