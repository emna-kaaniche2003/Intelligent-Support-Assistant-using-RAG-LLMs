"""
data_pipeline/detect_reclamations_redondantes.py
Détection des réclamations redondantes (clustering) pour le pipeline FAQ.

⚠️ Périmètre : clustering UNIQUEMENT (regroupement de réclamations similaires).
PAS de génération de question canonique, PAS de redirection support — sous-étape
suivante, traitée séparément. AUCUN appel LLM/Gemini/Vertex AI dans ce script :
embeddings locaux (sentence-transformers) + calcul pur numpy/Python.

Approche (validée) :
  reclamations_seules.csv
    → Phase 1 : bucketing par module_designation + mots-clés (module_keywords.json)
    → Phase 2 : fusion des doublons exacts (hash_doublon) intra-bucket
    → Phase 3 : embeddings sentence-transformers sur les représentants
    → Phase 4 : similarité cosinus + composantes connexes (union-find)
    → Phase 5 : mode dry-run (rapport de paires + scores)
    → Phase 6 : sortie finale (clusters + membres)

Seuils et modèle en configuration (ai_engine/config.py / .env), jamais hardcodés :
  - REDUNDANCY_SIMILARITY_THRESHOLD (regroupement, 0.90 validé sur dry-run réel)
  - REDUNDANCY_DRYRUN_FLOOR (affichage des paires en dry-run)
  - REDUNDANCY_MESSAGE_TRUNCATE (troncature message dans la formule d'embedding)
  - REDUNDANCY_MIN_INFORMATIVE_TOKENS (garde-fou d'information minimale, N=4
    validé par inspection manuelle — cf. count_informative_tokens)
  - embedding_model (réutilisé, cohérent avec ingest_qdrant.py)

⚠️ PIÈGE DÉJÀ RENCONTRÉ SUR LE GARDE-FOU D'INFORMATION MINIMALE (voir
count_informative_tokens/_strip_bucket_terms) : le retrait des termes du bucket
DOIT se faire en PHRASE ENTIÈRE, jamais en décomposant un terme multi-mots en
mots isolés — une première version décomposait "paramétrage état résultat" en
{paramétrage, état, résultat} et retirait "paramétrage" à tort d'un ticket
disant "paramétrage DU BILAN" (contenu différent, informatif), le faisant
paraître vide. Ne pas réintroduire cette décomposition mot-par-mot.

Usage :
    # dry-run : rapport de paires + scores pour caler le seuil (n'écrit PAS la
    # sortie finale)
    python data_pipeline/detect_reclamations_redondantes.py \\
        --input data_pipeline/clean/reclamations_seules.csv --dry-run

    # sortie finale (clusters) une fois le seuil validé
    python data_pipeline/detect_reclamations_redondantes.py \\
        --input data_pipeline/clean/reclamations_seules.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Constantes ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data_pipeline" / "clean" / "reclamations_seules.csv"
DEFAULT_KEYWORDS = REPO_ROOT / "data_pipeline" / "config" / "module_keywords.json"

# Colonnes attendues en entrée (le script complet en a besoin dès la sortie ;
# on échoue tôt si l'une manque).
REQUIRED_COLS = [
    "idMessage",
    "module_designation",
    "objet_clean",
    "message_clean",
    "texte_complet",
    "hash_doublon",
    "IdClient",
]

# Bucket réservé aux tickets qui ne matchent aucun mot-clé de leur module.
# JAMAIS forcé dans un autre bucket (décision de conception).
NON_CATEGORISE = "non_categorise"

# Seuil (en nombre de caractères) au-dessous duquel un mot-clé est matché par
# limite de mot (\bkw\b) plutôt qu'en sous-chaîne, pour éviter les faux matchs
# à l'intérieur d'un mot plus long (ex. "REG" dans "reglement", "TVA" dans un
# mot fabriqué). Décision : mots-clés de <= 4 caractères (PEC, TVA, HT, CNAM,
# REG, RIB...).
SHORT_KEYWORD_MAX_LEN = 4


# ── Normalisation (casse + accents) ───────────────────────────────────────────

def _fold(text: object) -> str:
    """
    Minuscule + suppression des accents (NFKD, retrait des diacritiques) +
    normalisation des espaces (runs -> un espace, strip), pour un matching
    module/mots-clés insensible à la casse, aux accents ET aux espaces
    superflus. Indispensable car les module_designation du CSV portent des
    espaces en fin ("Facturation  ", "Comptabilité ") issus de l'export SQL,
    et le texte aplati par clean_html() peut contenir des espaces multiples.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    nfkd = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", without_accents).strip().lower()


def _keyword_matches(keyword_folded: str, text_folded: str) -> bool:
    """
    True si `keyword_folded` (déjà normalisé) est présent dans `text_folded`.
    Mots-clés courts (<= SHORT_KEYWORD_MAX_LEN) : matching par limite de mot.
    Mots-clés longs : matching en sous-chaîne.
    """
    kw = keyword_folded.strip()
    if not kw:
        return False
    if len(kw) <= SHORT_KEYWORD_MAX_LEN:
        return re.search(rf"\b{re.escape(kw)}\b", text_folded) is not None
    return kw in text_folded


# ── Phase 0 : chargement ──────────────────────────────────────────────────────

def load_reclamations(path: Path) -> pd.DataFrame:
    """Charge reclamations_seules.csv (auto-détection séparateur ';'/',')."""
    df = None
    for sep in (",", ";"):
        try:
            candidate = pd.read_csv(path, encoding="utf-8-sig", sep=sep, low_memory=False)
            if len(candidate.columns) > 1:
                df = candidate
                break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"Impossible de lire {path} avec les séparateurs ',' ou ';'")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes obligatoires manquantes dans {path.name} : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )
    print(f"[info] {len(df)} réclamations chargées depuis {path.name}")
    return df.reset_index(drop=True)


def load_keywords(path: Path) -> dict:
    """Charge module_keywords.json (l'ordre des clés est préservé)."""
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if "modules" not in config:
        raise ValueError(f"'modules' absent de {path.name}")
    print(f"[info] dictionnaire mots-clés chargé : {list(config['modules'].keys())}")
    return config


# ── Phase 1 : bucketing ───────────────────────────────────────────────────────

def _build_term_index(kw_config: dict):
    """
    Prépare, à partir du dictionnaire :
      - folded_module_index : {module_folded -> nom module JSON}
      - module_sous_order    : {module JSON -> [sous_themes dans l'ordre JSON]}
      - sous_theme_terms     : {(module, sous_theme) -> [termes foldés]}
    keywords ET variantes sont fusionnés comme termes de matching (décision).
    """
    modules_cfg = kw_config.get("modules", {})
    folded_module_index = {_fold(m): m for m in modules_cfg}
    module_sous_order: dict[str, list[str]] = {}
    sous_theme_terms: dict[tuple[str, str], list[str]] = {}
    for module, subs in modules_cfg.items():
        module_sous_order[module] = list(subs.keys())
        for sous_theme, cfg in subs.items():
            terms = list(cfg.get("keywords", [])) + list(cfg.get("variantes", []))
            sous_theme_terms[(module, sous_theme)] = [_fold(t) for t in terms if t]
    return folded_module_index, module_sous_order, sous_theme_terms


def assign_buckets(df: pd.DataFrame, kw_config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Assigne à chaque ticket un sous_theme (ou NON_CATEGORISE) selon les mots-clés
    de SON module. Bucket = (module_designation, sous_theme).

    Départage si plusieurs sous-thèmes matchent : nombre de mots-clés matchés
    décroissant, puis ordre d'apparition dans le JSON (premier gagne à égalité).

    Retourne (df enrichi, rapport) où le rapport contient notamment le nombre de
    tickets ayant matché 2 sous-thèmes ou plus (fréquence des cas de départage).
    """
    folded_module_index, module_sous_order, sous_theme_terms = _build_term_index(kw_config)

    df = df.copy()
    sous_themes: list[str] = []
    n_matched_terms: list[int] = []
    n_matched_subthemes: list[int] = []

    for _, row in df.iterrows():
        text_folded = _fold(row["texte_complet"])
        module_key = folded_module_index.get(_fold(row["module_designation"]))

        if module_key is None:
            # module hors dictionnaire (ne devrait pas arriver sur le CSV FAQ,
            # filtré aux 3 modules) : jamais forcé, -> non_categorise.
            sous_themes.append(NON_CATEGORISE)
            n_matched_terms.append(0)
            n_matched_subthemes.append(0)
            continue

        scores: dict[str, int] = {}
        for sous_theme in module_sous_order[module_key]:
            terms = sous_theme_terms[(module_key, sous_theme)]
            count = sum(1 for t in terms if _keyword_matches(t, text_folded))
            if count > 0:
                scores[sous_theme] = count

        if not scores:
            sous_themes.append(NON_CATEGORISE)
            n_matched_terms.append(0)
            n_matched_subthemes.append(0)
            continue

        # départage : max count, puis ordre JSON (strict '>' + itération ordonnée)
        best_st, best_count = None, -1
        for sous_theme in module_sous_order[module_key]:
            if sous_theme in scores and scores[sous_theme] > best_count:
                best_st, best_count = sous_theme, scores[sous_theme]
        sous_themes.append(best_st)
        n_matched_terms.append(best_count)
        n_matched_subthemes.append(len(scores))

    df["sous_theme"] = sous_themes
    df["n_mots_cles_matches"] = n_matched_terms
    df["n_sous_themes_matches"] = n_matched_subthemes
    df["bucket_id"] = df["module_designation"].astype(str) + "::" + df["sous_theme"]

    report = {
        "n_tickets": len(df),
        "n_tickets_multi_sous_theme": int((df["n_sous_themes_matches"] >= 2).sum()),
    }
    return df, report


def build_bucket_report(df: pd.DataFrame) -> dict:
    """
    Construit un rapport de comptage par bucket (module, sous_theme), avec la
    part de non_categorise par module (sans jugement automatique — comptage brut).
    """
    counts = df.groupby(["module_designation", "sous_theme"]).size()
    counts_dict = {tuple(k): int(v) for k, v in counts.items()}

    per_module: dict[str, dict] = {}
    for module, grp in df.groupby("module_designation"):
        total = len(grp)
        n_non_cat = int((grp["sous_theme"] == NON_CATEGORISE).sum())
        per_module[str(module)] = {
            "total": total,
            "non_categorise": n_non_cat,
            "part_non_categorise": round(n_non_cat / total, 3) if total else 0.0,
        }

    return {
        "counts_par_bucket": counts_dict,
        "par_module": per_module,
        "n_tickets_multi_sous_theme": int((df["n_sous_themes_matches"] >= 2).sum()),
    }


# ── Phase 2 : fusion des doublons exacts (intra-bucket) ───────────────────────

def fuse_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe, DANS chaque bucket, les tickets partageant le même hash_doublon
    (déjà calculé en amont). Chaque groupe désigne un représentant = texte le
    plus long (texte_complet), départage par idMessage croissant.

    Ajoute :
      - exact_group_id        : identifiant du groupe exact (unique par bucket+hash)
      - exact_group_size      : nombre de tickets du groupe
      - is_representant_exact : True pour le représentant de chaque groupe

    Fusion STRICTEMENT intra-bucket : deux hash identiques dans deux buckets
    différents ne fusionnent pas.
    """
    df = df.copy().reset_index(drop=True)
    df["exact_group_id"] = ""
    df["exact_group_size"] = 0
    df["is_representant_exact"] = False

    text_len = df["texte_complet"].astype(str).str.len()

    for (bucket, h), idx in df.groupby(["bucket_id", "hash_doublon"]).groups.items():
        idx = list(idx)
        # représentant : texte le plus long, puis idMessage le plus petit
        rep = sorted(idx, key=lambda i: (-int(text_len.iloc[i]), df.iloc[i]["idMessage"]))[0]
        gid = f"{bucket}#{str(h)[:8]}"
        df.loc[idx, "exact_group_id"] = gid
        df.loc[idx, "exact_group_size"] = len(idx)
        df.loc[rep, "is_representant_exact"] = True

    return df


# ── Phase 3 : embeddings (sentence-transformers, local) ───────────────────────

# Nettoyage du bruit AVANT mesure de similarité, appliqué UNIQUEMENT à la
# construction de texte_pour_embedding — jamais aux colonnes stockées
# (objet_clean/message_clean/texte_complet restent intacts pour l'équipe support).
# Motivation : sur les tickets très courts (ex. "INTEGRATION CA BJR.VOILA LE
# PROBLEME"), l'embedding mesure surtout la similarité de la politesse partagée
# et des prénoms de collaborateurs cités, pas du problème réel — aucun seuil ne
# corrige ce bruit dans le signal. On le retire donc en amont.
#
# Liste FERMÉE (même esprit que _SIGNATURE_RE/_BUSINESS_REF_RE) :
#  - formules de salutation/clôture/politesse ;
#  - marqueurs génériques de collaborateurs, issus du suffixe système
#    ", à <collaborateur>" et de greetings synthétiques.
_EMBEDDING_NOISE_COLLABORATORS = [
    "collaborateur", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "eta", "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
    "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi",
    "psi", "omega",
]

# Alternatives ordonnées : phrases d'abord (longest-match), puis mots simples,
# puis noms. Accents tolérés sur les rares mots accentués (voil[aà], probl[eè]me,
# tr[eè]s). \b + IGNORECASE gèrent la ponctuation collée ("BJR.VOILA").
_EMBEDDING_NOISE_ALTERNATIVES = [
    # phrases
    r"[aà]\s+l['’]attention(?:\s+de)?",
    r"voil[aà]\s+le\s+probl[eè]me", r"voil[aà]\s+le\s+pb",
    r"voil[aà]\s+mon\s+probl[eè]me", r"voil[aà]\s+mon\s+pb",
    r"le\s+probl[eè]me", r"mon\s+probl[eè]me", r"le\s+pb", r"mon\s+pb",
    r"c['’]est\s+urgent", r"tr[eè]s\s+urgent",
    # mots de politesse / salutation
    r"bonjour", r"bjour", r"bjr", r"salut", r"slt", r"coucou",
    r"svp", r"stp", r"voil[aà]", r"voici", r"merci",
    r"urgente?", r"cordialement", r"salutations?",
    r"mlle", r"mme", r"monsieur", r"madame", r"mr", r"pb",
] + _EMBEDDING_NOISE_COLLABORATORS

_EMBEDDING_NOISE_RE = re.compile(
    r"(?i)\b(?:" + "|".join(_EMBEDDING_NOISE_ALTERNATIVES) + r")\b"
)
_ONLY_PUNCT_RE = re.compile(r"^[^\w]+$")


def _strip_embedding_noise(text: str) -> str:
    """
    Retire le bruit (politesse + noms de collaborateurs) d'un texte destiné à
    l'embedding. NE MODIFIE PAS les colonnes stockées : n'est appelée que par
    build_embedding_text(). Retire aussi les tokens résiduels purement ponctuation
    (ex. le "." laissé par "BJR.VOILA" une fois BJR et VOILA retirés).
    """
    if not isinstance(text, str) or not text:
        return "" if not isinstance(text, str) else text
    cleaned = _EMBEDDING_NOISE_RE.sub(" ", text)
    tokens = [t for t in cleaned.split() if not _ONLY_PUNCT_RE.match(t)]
    return " ".join(tokens).strip()


def build_embedding_text(objet_clean: object, message_clean: object, truncate: int) -> str:
    """
    Construit le texte d'embedding selon la formule validée :
      objet_clean + ". " + objet_clean + ". " + message_clean[:truncate]
    - Le sujet (objet_clean) est RÉPÉTÉ pour lui donner plus de poids sémantique
      (le sujet porte l'essentiel de l'intention de la réclamation).
    - message_clean est tronqué EXPLICITEMENT à `truncate` caractères pour éviter
      une troncature silencieuse par le modèle (paraphrase-multilingual-mpnet a
      une limite de ~128 tokens) et rendre la troncature reproductible/auditée.
    - Nettoyage du bruit (_strip_embedding_noise) appliqué à chaque composant
      APRÈS troncature : retire politesse + noms de collaborateurs qui gonflent
      artificiellement la similarité des tickets courts. Appliqué UNIQUEMENT ici,
      jamais aux colonnes stockées.
    - Repli : si objet_clean est vide (ou entièrement du bruit), on renvoie
      message_clean[:truncate] nettoyé seul (pas de préfixe ". . " parasite).
    """
    objet = _strip_embedding_noise(objet_clean.strip()) if isinstance(objet_clean, str) else ""
    message = message_clean if isinstance(message_clean, str) else ""
    msg_trunc = _strip_embedding_noise(message[:truncate])
    if not objet:
        return msg_trunc.strip()
    return f"{objet}. {objet}. {msg_trunc}".strip()


# ── Garde-fou "information minimale" (exclusion du clustering sémantique) ─────
#
# Sur des tickets très courts, même après retrait du bruit de politesse, il ne
# reste parfois QUE le mot qui a servi à assigner le bucket (ex. "INTEGRATION
# CA", "TQB", "B[TEL]") — mesurer une similarité cosinus entre deux tickets
# aussi vides revient à mesurer la similarité du bucket avec lui-même, pas du
# contenu réel. Ces tickets sont donc exclus du calcul de similarité (Phase 4),
# mais JAMAIS supprimés du corpus : ils sortent en cluster singleton avec
# methode_fusion="singleton_info_faible", pour signaler explicitement à la
# revue humaine qu'ils n'ont pas été comparés (faute de contenu), et non
# qu'ils ont été comparés puis jugés uniques.
#
# ⚠️ PIÈGE DÉJÀ RENCONTRÉ, NE PAS RÉINTRODUIRE : le retrait des termes du
# bucket doit se faire en PHRASE ENTIÈRE (substring de chaque terme complet),
# PAS en décomposant les termes multi-mots en mots isolés. Une première version
# décomposait "paramétrage état résultat" (variante etats_financiers) en mots
# séparés {paramétrage, état, résultat} et retirait chacun indépendamment —
# ce qui supprimait à tort "paramétrage" d'un ticket disant "paramétrage DU
# BILAN" (contenu réel et différent), le faisant paraître vide alors qu'il ne
# l'est pas. Toujours retirer les termes comme des phrases complètes.
#
# Seuil N=4, calé par inspection manuelle des tickets réels (pas un chiffre
# arbitraire) : à N=4, les 9 tickets exclus sont tous génuinement vides de
# contenu différenciant ("TQB", "INTEGRA", "B[TEL]", "LETTRAGE"...). À N=5,
# des tickets encore informatifs bien que concis (ex. "liste
# des immobilisations avec taux 100%", 25392 "code journal", ou la paire
# 6982/6992 qui ne diffère que par la politesse et serait un vrai doublon)
# se retrouveraient exclus à tort — régression de recall constatée et écartée.

def _strip_bucket_terms(text_folded: str, bucket_terms: list[str]) -> str:
    """
    Retire du texte (déjà foldé) les termes du bucket EN PHRASE ENTIÈRE (les
    plus longs d'abord, pour qu'une phrase soit retirée avant ses mots
    composants s'il y avait un chevauchement). Ne décompose jamais un terme
    multi-mots en mots isolés (cf. avertissement ci-dessus).
    """
    for term in sorted(set(bucket_terms), key=len, reverse=True):
        if term:
            text_folded = re.sub(re.escape(term), " ", text_folded)
    return text_folded


_INFORMATIVE_TOKEN_RE = re.compile(r"[a-zà-ÿ]{2,}")


def count_informative_tokens(embedding_text: str, bucket_terms: list[str]) -> int:
    """
    Compte les tokens informatifs de `embedding_text` (déjà passé par
    _strip_embedding_noise via build_embedding_text) : alphabétiques, >= 2
    lettres, après retrait des termes du bucket en phrase entière. Aucun
    filtrage de stopwords grammaticaux supplémentaire (le/la/ne/pas...) — vue
    volontairement simple, décision confirmée sur inspection des cas réels.
    """
    stripped = _strip_bucket_terms(_fold(embedding_text), bucket_terms)
    return len(_INFORMATIVE_TOKEN_RE.findall(stripped))


def encode_texts(texts: list[str], model_name: str):
    """
    Encode les textes en vecteurs denses normalisés L2 (cosinus = produit
    scalaire), en un seul batch. sentence-transformers en LOCAL — AUCUN appel
    LLM/Vertex AI. Modèle réutilisé depuis la config (cohérent ingest_qdrant.py).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not texts:
        return np.zeros((0, 0))
    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


# ── Phase 4 : similarité cosinus + composantes connexes (union-find) ──────────

def cosine_matrix(vectors):
    """Matrice de similarité cosinus (vecteurs supposés normalisés L2)."""
    import numpy as np
    v = np.asarray(vectors, dtype=float)
    if v.size == 0:
        return np.zeros((0, 0))
    return v @ v.T


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """
    Composantes connexes (union-find) sur n nœuds et une liste d'arêtes.
    Regroupement single-linkage : A~B et B~C ⇒ {A,B,C} même si A≁C. Comportement
    assumé et documenté (pas de clustering ML, décision de conception).
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a, b in edges:
        union(a, b)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    # ordonné par plus petit membre, pour un résultat déterministe
    return [comps[k] for k in sorted(comps)]


def cluster_indices(vectors, threshold: float) -> list[list[int]]:
    """
    Regroupe les vecteurs d'un bucket : arête entre i et j si cosinus >= seuil,
    puis composantes connexes. Retourne des listes d'indices LOCAUX (0..n-1).
    """
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    sim = cosine_matrix(vectors)
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if sim[i, j] >= threshold
    ]
    return connected_components(n, edges)


# ── Phase 5 : dry-run (rapport de paires + scores) ────────────────────────────

def build_dryrun_pairs(reps_df: pd.DataFrame, vectors, floor: float) -> pd.DataFrame:
    """
    Pour chaque bucket, liste TOUTES les paires de représentants dont le cosinus
    est >= `floor`, triées par score décroissant (par bucket). reps_df doit avoir
    un RangeIndex 0..n-1 aligné sur les lignes de `vectors`.

    Les représentants "is_low_info" (garde-fou d'information minimale, si la
    colonne est présente) sont EXCLUS : jamais comparés, donc jamais dans le
    rapport de paires, cohérent avec la sortie finale (singleton_info_faible).
    """
    sim = cosine_matrix(vectors)
    rows = []
    if "is_low_info" in reps_df.columns:
        reps_df = reps_df[~reps_df["is_low_info"]]
    for bucket, grp in reps_df.groupby("bucket_id"):
        pos = list(grp.index)
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                i, j = pos[a], pos[b]
                score = float(sim[i, j])
                if score >= floor:
                    rows.append({
                        "bucket_id": bucket,
                        "module_designation": reps_df.at[i, "module_designation"],
                        "sous_theme": reps_df.at[i, "sous_theme"],
                        "idMessage_a": reps_df.at[i, "idMessage"],
                        "idMessage_b": reps_df.at[j, "idMessage"],
                        "cosinus": round(score, 4),
                        "texte_a": reps_df.at[i, "texte_complet"],
                        "texte_b": reps_df.at[j, "texte_complet"],
                    })
    cols = ["bucket_id", "module_designation", "sous_theme", "idMessage_a",
            "idMessage_b", "cosinus", "texte_a", "texte_b"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["bucket_id", "cosinus"], ascending=[True, False]).reset_index(drop=True)


# ── Phase 6 : assemblage des clusters + sortie ────────────────────────────────

def _distinct_clients(members: pd.DataFrame) -> int:
    """Nombre d'IdClient distincts, en IGNORANT les valeurs vides/NA (décision)."""
    vals = members["IdClient"].dropna()
    vals = vals[vals.astype(str).str.strip() != ""]
    return int(vals.nunique())


def assemble_clusters(fused_df: pd.DataFrame, reps_df: pd.DataFrame, vectors,
                      threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construit les clusters finaux à partir des représentants (post fusion exacte)
    et de leurs embeddings. reps_df : RangeIndex 0..n-1 aligné sur `vectors`.

    methode_fusion :
      - "semantique"          : ≥2 représentants regroupés par similarité cosinus
      - "exact"               : 1 représentant (comparable ou non), mais son
                                 groupe exact contient ≥2 tickets (hash_doublon)
      - "singleton"           : 1 représentant, 1 seul ticket, comparable mais
                                 sans voisin au-dessus du seuil
      - "singleton_info_faible" : représentant EXCLU du calcul de similarité
                                 (garde-fou d'information minimale, colonne
                                 is_low_info) ET sans duplicata exact — jamais
                                 comparé, pas juste "jugé unique"

    Les représentants is_low_info ne sont JAMAIS passés à cluster_indices : ils
    ne peuvent donc avoir aucune arête, ni entre eux ni vers un représentant
    comparable — exclusion garantie, pas une histoire de seuil.

    Retourne (clusters_df, membres_df).
    """
    sim = cosine_matrix(vectors)
    text_len_all = fused_df["texte_complet"].astype(str).str.len()
    has_low_info_col = "is_low_info" in reps_df.columns

    cluster_rows = []
    membre_rows = []
    counters: dict[str, int] = {}

    for bucket, grp in reps_df.groupby("bucket_id"):
        pos = list(grp.index)
        if has_low_info_col:
            comparable_pos = [p for p in pos if not reps_df.at[p, "is_low_info"]]
            low_info_pos = [p for p in pos if reps_df.at[p, "is_low_info"]]
        else:
            comparable_pos, low_info_pos = pos, []

        local_vecs = vectors[comparable_pos] if comparable_pos else vectors[:0]
        semantic_comps = cluster_indices(local_vecs, threshold)
        # composantes en positions GLOBALES (reps_df/vectors), + un singleton
        # isolé par représentant low-info (jamais fusionné sémantiquement)
        comps = [[comparable_pos[c] for c in comp] for comp in semantic_comps]
        comps += [[p] for p in low_info_pos]

        for rep_positions in comps:
            module = reps_df.at[rep_positions[0], "module_designation"]
            sous_theme = reps_df.at[rep_positions[0], "sous_theme"]

            # tous les membres = union des groupes exacts de chaque représentant
            gids = [reps_df.at[p, "exact_group_id"] for p in rep_positions]
            members = fused_df[fused_df["exact_group_id"].isin(gids)]

            # représentant du cluster = texte_complet le plus long parmi tous
            member_idx = list(members.index)
            cluster_rep_idx = sorted(
                member_idx,
                key=lambda i: (-int(text_len_all.iloc[i]), fused_df.iloc[i]["idMessage"]),
            )[0]
            cluster_rep_id = fused_df.iloc[cluster_rep_idx]["idMessage"]
            # position du représentant du cluster dans reps_df/vectors (c'est un
            # représentant exact, donc présent dans reps_df)
            cluster_rep_pos = reps_df.index[reps_df["idMessage"] == cluster_rep_id][0]

            # méthode
            is_low_info_rep = (
                len(rep_positions) == 1
                and has_low_info_col
                and bool(reps_df.at[rep_positions[0], "is_low_info"])
            )
            if len(rep_positions) > 1:
                methode = "semantique"
            elif int(members["exact_group_size"].iloc[0]) > 1:
                methode = "exact"
            elif is_low_info_rep:
                methode = "singleton_info_faible"
            else:
                methode = "singleton"

            # score min intra-cluster (paires de représentants) pour semantique
            if len(rep_positions) > 1:
                pair_scores = [
                    float(sim[a, b])
                    for ia, a in enumerate(rep_positions)
                    for b in rep_positions[ia + 1:]
                ]
                score_min = round(min(pair_scores), 4) if pair_scores else ""
            else:
                score_min = ""

            counters[f"{module}::{sous_theme}"] = counters.get(f"{module}::{sous_theme}", 0) + 1
            cluster_id = f"{module}_{sous_theme}_{counters[f'{module}::{sous_theme}']}"

            member_ids = members["idMessage"].tolist()
            cluster_rows.append({
                "cluster_id": cluster_id,
                "module": module,
                "sous_theme": sous_theme,
                "methode_fusion": methode,
                "nb_tickets": len(member_ids),
                "nb_clients_distincts": _distinct_clients(members),
                "idMessages_membres": ";".join(str(i) for i in member_ids),
                "idMessage_representant": cluster_rep_id,
                "score_similarite_min": score_min,
                "texte_representant": fused_df.iloc[cluster_rep_idx]["texte_complet"],
            })

            # membres : cosinus au représentant du cluster (via le représentant
            # exact de chaque membre ; les doublons exacts héritent du leur)
            rep_pos_by_gid = {reps_df.at[p, "exact_group_id"]: p for p in rep_positions}
            for _, m in members.iterrows():
                m_rep_pos = rep_pos_by_gid[m["exact_group_id"]]
                cos = 1.0 if m_rep_pos == cluster_rep_pos else round(float(sim[m_rep_pos, cluster_rep_pos]), 4)
                membre_rows.append({
                    "idMessage": m["idMessage"],
                    "cluster_id": cluster_id,
                    "methode_fusion": methode,
                    "cosinus_au_representant": cos,
                })

    clusters_df = pd.DataFrame(cluster_rows)
    membres_df = pd.DataFrame(membre_rows)
    return clusters_df, membres_df


# ── Orchestration (testable, encode_fn injectable) ────────────────────────────

def _module_key_for(row_module: str, folded_module_index: dict) -> str | None:
    return folded_module_index.get(_fold(row_module))


def prepare_representatives(
    fused_df: pd.DataFrame,
    truncate: int,
    kw_config: dict,
    min_informative_tokens: int,
) -> pd.DataFrame:
    """
    Représentants post-fusion exacte, RangeIndex 0..n-1, + texte d'embedding +
    garde-fou d'information minimale (n_tokens_informatifs / is_low_info).
    """
    _, _, sous_theme_terms = _build_term_index(kw_config)
    folded_module_index = {_fold(m): m for m in kw_config.get("modules", {})}

    reps = fused_df[fused_df["is_representant_exact"]].reset_index(drop=True).copy()
    reps["texte_pour_embedding"] = [
        build_embedding_text(o, m, truncate)
        for o, m in zip(reps["objet_clean"], reps["message_clean"])
    ]

    n_tokens = []
    for _, row in reps.iterrows():
        module_key = _module_key_for(row["module_designation"], folded_module_index)
        terms = sous_theme_terms.get((module_key, row["sous_theme"]), [])
        n_tokens.append(count_informative_tokens(row["texte_pour_embedding"], terms))
    reps["n_tokens_informatifs"] = n_tokens
    reps["is_low_info"] = reps["n_tokens_informatifs"] < min_informative_tokens
    return reps


def run_redundancy_detection(
    df: pd.DataFrame,
    kw_config: dict,
    *,
    threshold: float,
    dryrun_floor: float,
    truncate: int,
    model_name: str,
    dry_run: bool,
    output_dir: Path,
    min_informative_tokens: int = 0,
    encode_fn=encode_texts,
):
    """
    Enchaîne Phases 1→ (4|6). En dry-run : écrit uniquement le rapport de paires
    et NE crée PAS la sortie finale. Retourne un dict de résultats en mémoire
    (pour tests et reporting).
    """
    bucketed, _ = assign_buckets(df, kw_config)
    fused = fuse_exact_duplicates(bucketed)
    reps = prepare_representatives(fused, truncate, kw_config, min_informative_tokens)

    vectors = encode_fn(list(reps["texte_pour_embedding"]), model_name)

    result = {"fused": fused, "representatives": reps, "vectors": vectors}

    if dry_run:
        pairs = build_dryrun_pairs(reps, vectors, dryrun_floor)
        output_dir.mkdir(parents=True, exist_ok=True)
        dryrun_path = output_dir / "reclamations_redondantes_dryrun.csv"
        pairs.to_csv(dryrun_path, index=False, encoding="utf-8-sig")
        result["dryrun_pairs"] = pairs
        result["dryrun_path"] = dryrun_path
        return result

    clusters_df, membres_df = assemble_clusters(fused, reps, vectors, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_path = output_dir / "reclamations_redondantes.csv"
    membres_path = output_dir / "reclamations_redondantes_membres.csv"
    clusters_df.to_csv(clusters_path, index=False, encoding="utf-8-sig")
    membres_df.to_csv(membres_path, index=False, encoding="utf-8-sig")
    result["clusters"] = clusters_df
    result["membres"] = membres_df
    result["clusters_path"] = clusters_path
    result["membres_path"] = membres_path
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_bucket_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 1 — COMPTAGE PAR BUCKET (module, sous_theme)")
    print("=" * 70)
    counts = report["counts_par_bucket"]
    for module in sorted({m for (m, _st) in counts}):
        module_buckets = {st: n for (m, st), n in counts.items() if m == module}
        total = sum(module_buckets.values())
        print(f"\n  {module}  (total {total})")
        for st in sorted(module_buckets, key=lambda s: (-module_buckets[s], s)):
            flag = "  <-- non catégorisé" if st == NON_CATEGORISE else ""
            print(f"      {module_buckets[st]:>4}  {st}{flag}")
    print("\n  " + "-" * 66)
    for module, stats in report["par_module"].items():
        print(
            f"  {module:<14} non_categorise = {stats['non_categorise']}/{stats['total']} "
            f"({stats['part_non_categorise'] * 100:.1f} %)"
        )
    print(f"\n  Tickets ayant matché 2 sous-thèmes ou plus (cas de départage) : "
          f"{report['n_tickets_multi_sous_theme']}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détection des réclamations redondantes (bucketing + fusion exacte + clustering)"
    )
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--keywords", "-k", type=Path, default=DEFAULT_KEYWORDS)
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_INPUT.parent)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Écrit uniquement le rapport de paires (scores) pour caler le seuil ; "
             "n'écrit PAS la sortie finale des clusters.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[erreur] Fichier introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.keywords.exists():
        print(f"[erreur] Dictionnaire introuvable : {args.keywords}", file=sys.stderr)
        sys.exit(1)

    from ai_engine.config import get_settings
    cfg = get_settings()

    df = load_reclamations(args.input)
    kw_config = load_keywords(args.keywords)

    # Phases 1-2 : comptage + fusion exacte (affichage)
    bucketed, _ = assign_buckets(df, kw_config)
    _print_bucket_report(build_bucket_report(bucketed))
    fused = fuse_exact_duplicates(bucketed)
    n_reps = int(fused["is_representant_exact"].sum())
    print("\n" + "=" * 70)
    print("  PHASE 2 — FUSION DES DOUBLONS EXACTS (intra-bucket)")
    print("=" * 70)
    print(f"  {len(fused)} tickets -> {n_reps} représentants "
          f"({len(fused) - n_reps} doublons exacts fusionnés)")

    # Phases 3-(4|6)
    print(f"\n[info] Modèle d'embedding : {cfg.embedding_model} (local, aucun appel LLM)")
    print(f"[info] Seuil={cfg.redundancy_similarity_threshold} | "
          f"plancher dry-run={cfg.redundancy_dryrun_floor} | "
          f"troncature message={cfg.redundancy_message_truncate} | "
          f"min tokens informatifs={cfg.redundancy_min_informative_tokens}")

    result = run_redundancy_detection(
        df, kw_config,
        threshold=cfg.redundancy_similarity_threshold,
        dryrun_floor=cfg.redundancy_dryrun_floor,
        truncate=cfg.redundancy_message_truncate,
        model_name=cfg.embedding_model,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        min_informative_tokens=cfg.redundancy_min_informative_tokens,
    )

    if args.dry_run:
        pairs = result["dryrun_pairs"]
        print("\n" + "=" * 70)
        print("  PHASE 5 — DRY-RUN (paires de représentants, cosinus >= plancher)")
        print("=" * 70)
        print(f"  {len(pairs)} paires >= {cfg.redundancy_dryrun_floor} écrites dans {result['dryrun_path']}")
        print("  (sortie finale des clusters NON écrite — dry-run)")
    else:
        print("\n" + "=" * 70)
        print("  PHASE 6 — SORTIE FINALE")
        print("=" * 70)
        print(f"  {len(result['clusters'])} clusters -> {result['clusters_path']}")
        print(f"  {len(result['membres'])} lignes membres -> {result['membres_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
