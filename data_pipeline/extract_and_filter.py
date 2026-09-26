"""
data_pipeline/extract_and_filter.py
Nettoyage et filtrage des tickets bruts avant ingestion dans Qdrant.

Pipeline :
  raw/tickets_bruts.csv
    → Étape 0 : chargement + mapping colonnes
    → Étape 1 : nettoyage HTML + règles (NaN, doublons, longueur minimale)
    → Étape 2 : filtre LLM HuggingFace (désactivable avec --no-llm)
    → clean/reclamations_et_reponses.csv  +  clean/reclamations_et_reponses_filter_report.csv

Usage :
    python data_pipeline/extract_and_filter.py \\
        --input data_pipeline/raw/tickets_bruts.csv \\
        --col-map titre=objet,description=message,resolution=observation

    python data_pipeline/extract_and_filter.py \\
        --input data_pipeline/raw/tickets_bruts.csv \\
        --col-map titre=objet,description=message,resolution=observation \\
        --no-llm
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_engine.config import get_settings
from data_pipeline.cleaning_utils import clean_html, hash_text, is_noise

# Forcer UTF-8 sur stdout/stderr pour éviter les erreurs cp1252 (données arabes)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Constantes ────────────────────────────────────────────────────────────────

REQUIRED_COLS = ["titre", "description", "resolution"]
OPTIONAL_COLS = ["ticket_id", "priorite", "module", "type_priorite", "statut", "date_creation"]

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "ai_engine" / "prompts"
FILTER_PROMPT_FILE = PROMPTS_DIR / "filter_ticket.txt"

OUTPUT_DIR = Path(__file__).resolve().parent / "clean"

# Patterns de contenu sensible — si trouvé dans titre/description/resolution → rejet
SECURITY_PATTERNS = [
    r"(?i)(password|mot\s+de\s+passe|mdp|pass)\s*[:=]\s*\S+",
    r"(?i)(login|username|identifiant|compte)\s*[:=]\s*\S+",
    r"(?i)P@[s$][s$]w?\w+",                              # P@ssw0rd, P@$sw0rd
    r"(?i)(hyper[\s\-]?v|sql[\s\-]?server|fire[\s\-]?wall|hmail)\s*[:=]\s*\S+",
    r"(?i)(clé\s+(api|ssh|privée?)|token|secret)\s*[:=]\s*\S+",
]

# Patterns de résolution vague — aucune valeur pour le RAG
VAGUE_RESOLUTION_PATTERNS = [
    r"(?i)^[\w\s,]{0,30}(tâche|tache)\s+achev[ée]s?[.\s]*$",
    r"(?i)^[\w\s,]{0,40}bien\s+traité[e]?[.\s]*$",
    r"(?i)^[\w\s,]{0,50}(assistance|intervention)\s+(effectu[ée]|réalis[ée]|faite?|terminé[e]?)[.\s]*$",
    r"(?i)^[\w\s,]{0,40}réclamation\s+(bien\s+)?(traitée?|résolue?)[.\s]*$",
    r"(?i)^[\w\s,]{0,40}(problème|pb|issue)\s+(résolu[e]?|réglé[e]?|corrigé[e]?)[.\s]*$",
    r"(?i)^[\w\s,]{0,20}(résolu[e]?|réglé[e]?|traité[e]?|corrigé[e]?|terminé[e]?|done|ok|fait)[.\s!]*$",
    r"(?i)^[\w\s,]{0,30}(mission\s+accomplie|travail\s+(effectué|réalisé|terminé))[.\s]*$",
]


# ── Chargement ────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # Auto-détection séparateur : essaye ; puis ,
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, encoding="utf-8-sig", sep=sep, low_memory=False)
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError(f"Format non supporté : {suffix}. Utiliser .csv ou .xlsx")

    # Supprimer les colonnes vides générées par l'export (Unnamed:X)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [c.strip() for c in df.columns]

    print(f"[info] {len(df)} lignes chargées depuis {path.name}")
    print(f"[info] Colonnes détectées : {list(df.columns)}")
    if len(df) > 0:
        sample = {k: str(v)[:80] for k, v in df.iloc[0].items()}
        print(f"[info] Exemple ligne 0 : {sample}")
    return df


def apply_col_map(df: pd.DataFrame, col_map: Optional[str]) -> pd.DataFrame:
    """
    Renomme les colonnes selon --col-map titre=X,description=Y,resolution=Z.
    Si col_map est None, vérifie que les colonnes cibles existent déjà.
    """
    if col_map:
        mapping: dict[str, str] = {}
        for pair in col_map.split(","):
            pair = pair.strip()
            if "=" not in pair:
                print(f"[warn] Paire ignorée (format invalide) : '{pair}'", file=sys.stderr)
                continue
            target, source = pair.split("=", 1)
            mapping[source.strip()] = target.strip()
        df = df.rename(columns=mapping)
        print(f"[info] Mapping appliqué : {mapping}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(
            f"[erreur] Colonnes obligatoires manquantes : {missing}\n"
            f"         Utilise --col-map titre=X,description=Y,resolution=Z\n"
            f"         Colonnes disponibles : {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    cols_to_keep = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df.columns]
    df = df[cols_to_keep].copy()

    # Correctif à la source : strip des colonnes catégorielles optionnelles
    # stockées (module/priorite/type_priorite/statut). L'export SQL laisse des
    # espaces en fin (ex. "Facturation  ") qui casseraient tout matching aval
    # par égalité — même classe de bug que dans le pipeline FAQ. On conserve
    # casse/accents (strip + runs d'espaces -> un espace), NA laissés tels quels.
    for col in ("module", "priorite", "type_priorite", "statut"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v
            )
    return df


# ── Étape 1 : nettoyage HTML + règles ────────────────────────────────────────

def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return re.sub(r"\s+", " ", text.strip())


def has_security_content(row: pd.Series) -> bool:
    """Retourne True si le ticket contient des données sensibles (mdp, credentials, tokens)."""
    text = " ".join([
        str(row.get("titre", "")),
        str(row.get("description", "")),
        str(row.get("resolution", "")),
    ])
    return any(re.search(p, text) for p in SECURITY_PATTERNS)


def is_vague_resolution(text: str) -> bool:
    """Retourne True si la résolution est une phrase générique sans valeur pour le RAG."""
    # Nettoyer les formules de politesse courantes avant de tester
    cleaned = re.sub(r"(?i)bonjour[\s,]*", "", text).strip()
    cleaned = re.sub(r"(?i)cordialement[\s,\w]*\.?\s*$", "", cleaned).strip()
    return any(re.search(p, cleaned) for p in VAGUE_RESOLUTION_PATTERNS)


def apply_cleaning_and_rules(
    df: pd.DataFrame,
    min_desc: int,
    min_res: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1. Nettoie le HTML sur titre/description/resolution
    2. Applique les règles de filtrage
    Retourne (tickets_retenus, rapport_rejetés).
    """
    df = df.copy()

    # --- Nettoyage HTML sur les colonnes texte ---
    print("[info] Nettoyage HTML en cours…")
    for col in REQUIRED_COLS:
        df[col] = df[col].apply(clean_html)

    reject_mask = pd.Series(False, index=df.index)
    reasons: dict[int, str] = {}

    # 1. Champs vides après nettoyage HTML
    for col in REQUIRED_COLS:
        mask = df[col].apply(lambda t: is_noise(t, 1))
        for idx in df[mask & ~reject_mask].index:
            reasons[idx] = f"champ '{col}' vide après nettoyage HTML"
        reject_mask |= mask

    # 2. Longueur minimale description
    mask = df["description"].apply(lambda t: is_noise(t, min_desc))
    for idx in df[mask & ~reject_mask].index:
        reasons[idx] = f"description trop courte (< {min_desc} chars)"
    reject_mask |= mask

    # 3. Longueur minimale résolution
    mask = df["resolution"].apply(lambda t: is_noise(t, min_res))
    for idx in df[mask & ~reject_mask].index:
        reasons[idx] = f"résolution trop courte (< {min_res} chars)"
    reject_mask |= mask

    # 4. Déduplications par hash description
    hashes = df["description"].apply(hash_text)
    dup_mask = hashes.duplicated(keep="first")
    for idx in df[dup_mask & ~reject_mask].index:
        reasons[idx] = "doublon (description identique)"
    reject_mask |= dup_mask

    # 5. Contenu sensible (mots de passe, credentials, tokens)
    sec_mask = df.apply(has_security_content, axis=1)
    for idx in df[sec_mask & ~reject_mask].index:
        reasons[idx] = "contenu sensible détecté (mdp/credentials/token)"
    reject_mask |= sec_mask

    # 6. Résolution vague (aucune valeur pour le RAG)
    vague_mask = df["resolution"].apply(is_vague_resolution)
    for idx in df[vague_mask & ~reject_mask].index:
        reasons[idx] = "résolution vague (non actionnable pour le RAG)"
    reject_mask |= vague_mask

    kept = df[~reject_mask].reset_index(drop=True)
    rejected = df[reject_mask].copy()
    rejected["raison_rejet"] = rejected.index.map(reasons)
    rejected["statut_filtre"] = "rejeté_règles"
    rejected = rejected.reset_index(drop=True)

    n_total = len(df)
    n_sec = int(sec_mask.sum())
    n_vague = int(vague_mask.sum())
    print(
        f"[règles] {n_total} → {len(kept)} retenus "
        f"({n_total - len(kept)} rejetés : vides/courts/doublons"
        f"{f', {n_sec} sécurité' if n_sec else ''}"
        f"{f', {n_vague} résolution vague' if n_vague else ''})"
    )
    return kept, rejected


# ── Étape 2 : filtre LLM ─────────────────────────────────────────────────────

def load_prompt_template() -> str:
    if not FILTER_PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier prompt introuvable : {FILTER_PROMPT_FILE}\n"
            "Vérifier que ai_engine/prompts/filter_ticket.txt existe."
        )
    return FILTER_PROMPT_FILE.read_text(encoding="utf-8")


def call_llm(client, model: str, prompt: str) -> dict:
    """
    Appelle Gemini et parse la réponse JSON.
    Fail-safe : si JSON invalide → conserve le ticket (pertinent=True).
    """
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        raw = response.text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"pertinent": True, "raison": f"JSON non parseable : {raw[:80]}"}
    except Exception as exc:
        return {"pertinent": True, "raison": f"erreur LLM (fail-safe conservé) : {exc}"}


def apply_llm_filter(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retourne (tickets_retenus, tickets_rejetés_par_llm).
    """
    from google import genai
    from google.oauth2 import service_account

    cfg = get_settings()
    credentials = service_account.Credentials.from_service_account_file(
        cfg.google_application_credentials,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(
        vertexai=True,
        project=cfg.gemini_project,
        location=cfg.gemini_location,
        credentials=credentials,
    )
    prompt_template = load_prompt_template()

    kept_indices: list[int] = []
    rejected_rows: list[dict] = []

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="LLM filter")):
        prompt = prompt_template.format(
            titre=row["titre"],
            description=row["description"],
            resolution=row["resolution"],
        )
        result = call_llm(client, cfg.gemini_model, prompt)

        if result.get("pertinent", True):
            kept_indices.append(i)
        else:
            rejected_rows.append({
                **row.to_dict(),
                "statut_filtre": "rejeté_LLM",
                "raison_rejet": result.get("raison", ""),
            })

    kept = df.iloc[kept_indices].reset_index(drop=True)
    rejected_llm = pd.DataFrame(rejected_rows)

    print(
        f"[LLM]   {len(df)} → {len(kept)} retenus "
        f"({len(df) - len(kept)} rejetés par le LLM)"
    )
    return kept, rejected_llm


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_outputs(
    kept: pd.DataFrame,
    rejected_rules: pd.DataFrame,
    rejected_llm: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = OUTPUT_DIR / "reclamations_et_reponses.csv"
    kept.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"[ok] reclamations_et_reponses.csv → {clean_path} ({len(kept)} tickets)")

    all_rejected = pd.concat([rejected_rules, rejected_llm], ignore_index=True)
    all_kept = kept.copy()
    all_kept["statut_filtre"] = "retenu"
    all_kept["raison_rejet"] = ""
    report = pd.concat([all_kept, all_rejected], ignore_index=True)

    report_path = OUTPUT_DIR / "reclamations_et_reponses_filter_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"[ok] reclamations_et_reponses_filter_report.csv → {report_path} ({len(report)} lignes total)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nettoyage HTML + filtrage des tickets bruts avant ingestion RAG"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=Path,
        help="Chemin vers le fichier brut (CSV ou XLSX)",
    )
    parser.add_argument(
        "--col-map",
        default=None,
        help="Mapping colonnes : titre=X,description=Y,resolution=Z",
    )
    parser.add_argument(
        "--min-desc",
        type=int,
        default=80,
        help="Longueur minimale de la description après nettoyage (défaut : 80)",
    )
    parser.add_argument(
        "--min-res",
        type=int,
        default=100,
        help="Longueur minimale de la résolution après nettoyage (défaut : 100)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Sauter l'étape de filtrage LLM (règles uniquement)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"[erreur] Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    # Étape 0 — Chargement + mapping colonnes
    df = load_raw(args.input)
    n_bruts = len(df)
    df = apply_col_map(df, args.col_map)

    # Étape 1 — Nettoyage HTML + règles
    df_kept, df_rejected_rules = apply_cleaning_and_rules(df, args.min_desc, args.min_res)

    # Étape 2 — LLM (optionnel)
    df_rejected_llm = pd.DataFrame()
    if args.no_llm:
        print("[info] Étape LLM ignorée (--no-llm)")
    else:
        df_kept, df_rejected_llm = apply_llm_filter(df_kept)

    # Étape 3 — Sauvegarde
    save_outputs(df_kept, df_rejected_rules, df_rejected_llm)

    print(
        f"\n[résumé] {n_bruts} bruts "
        f"→ {len(df_kept)} retenus pour Qdrant "
        f"({n_bruts - len(df_kept)} rejetés au total)"
    )
    print(
        "[suivant] python data_pipeline/ingest_qdrant.py "
        "--input data_pipeline/clean/reclamations_et_reponses.csv"
    )


if __name__ == "__main__":
    main()
