"""
test_keyword_expansion_comparison.py
Compare 3 sources de rang pour un sous-ensemble de questions du golden dataset :
  1. BASELINE  : rang sans expansion (déjà mesuré, ré-utilisé depuis les rapports existants)
  2. ORACLE    : rang avec mots_cles du point cible concaténés (déjà mesuré, ré-utilisé)
  3. GÉNÉRÉ    : rang avec mots-clés générés par un LLM à partir de la seule question
     client (nouveau calcul, source : tests/keyword_expansion_generated_v1.json)

Script ISOLÉ, lecture seule sur golden_dataset.json et sur Qdrant. Ne modifie ni
evaluate_retrieval.py, ni les points Qdrant, ni test_keyword_expansion.py.
Réutilise (n'importe, ne duplique pas) dense_only_search/hybrid_search/
reciprocal_rank_fusion depuis ai_engine/rag/evaluate_retrieval.py.

Usage :
    python test_keyword_expansion_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ai_engine.config import get_settings
from ai_engine.rag.evaluate_retrieval import (
    DEFAULT_GOLDEN_PATH,
    DEFAULT_INGESTION_CONFIG_PATH,
    build_reference_corpus,
    dense_only_search,
    hybrid_search,
    load_golden_dataset,
    load_ingestion_config,
)
from data_pipeline.ingestion.ingest_qr_technique import (
    build_bm25_index,
    build_point_id,
    tokenize,
    vectorize_bm25,
)

GENERATED_KEYWORDS_PATH = "tests/keyword_expansion_generated_v1.json"
DENSE_ONLY_REPORT_PATH = "tests/retrieval_eval_dense_only_report.json"
HYBRID_REPORT_PATH = "tests/retrieval_eval_report.json"
COMPARISON_REPORT_PATH = "tests/keyword_expansion_comparison_report.json"

INSPECTION_LIMIT = 20  # même profondeur que test_keyword_expansion.py, pour comparabilité directe


# ── Étape 1 : chargement + validation stricte (ne rien supposer) ────────────

def load_generated_keywords_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_generated_entries(raw: Any) -> list[dict[str, Any]]:
    """
    Le format attendu documenté par l'utilisateur est :
    {"generated_by": ..., "prompt_source": ..., "results": [{"id_test": ...,
    "mots_cles_generes": [...]}]}.
    On ne suppose PAS ce format silencieusement : on vérifie sa présence et on
    échoue explicitement s'il ne correspond pas à ce qui est documenté.
    """
    if not isinstance(raw, dict) or "results" not in raw:
        raise ValueError(
            "Format de fichier invalide : attendu un objet JSON avec une clé "
            "'results' (et idéalement 'generated_by'/'prompt_source'), reçu : "
            f"{type(raw).__name__}"
            + (f" avec clés {list(raw.keys())}" if isinstance(raw, dict) else
               f" (liste de {len(raw)} éléments)" if isinstance(raw, list) else "")
        )
    return raw["results"]


def validate_entries_against_golden(
    generated_entries: list[dict[str, Any]],
    golden_questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """
    Valide chaque entrée du fichier généré contre golden_dataset.json.

    Ne se contente PAS de vérifier que l'id_test existe comme clé (un même
    id_test pourrait exister avec un contenu totalement différent dans un
    fichier mal aligné) : croise aussi question_client pour détecter les
    collisions de nom sans correspondance de contenu réel.

    Retourne (entrées_valides, id_tests_orphelins, entrées_avec_contenu_incohérent).
    """
    golden_by_id = {q["id_test"]: q for q in golden_questions}

    valides: list[dict[str, Any]] = []
    orphelins: list[str] = []
    incoherentes: list[dict[str, Any]] = []

    for entry in generated_entries:
        id_test = entry.get("id_test")
        if id_test not in golden_by_id:
            orphelins.append(id_test)
            continue

        golden_q = golden_by_id[id_test]
        mots_cles = entry.get("mots_cles_generes")
        if mots_cles is None:
            incoherentes.append({
                "id_test": id_test,
                "reason": "champ 'mots_cles_generes' absent de l'entrée",
            })
            continue

        # Garde-fou anti-collision : même id_test mais question différente
        # du golden dataset réel = fichier mal aligné, pas juste un orphelin.
        entry_question = entry.get("question_client")
        golden_question = golden_q.get("question_client")
        if entry_question is not None and entry_question != golden_question:
            incoherentes.append({
                "id_test": id_test,
                "reason": "question_client de l'entrée diffère de celle du golden dataset (même id_test, contenu différent)",
                "question_golden_dataset": golden_question,
                "question_fichier_genere": entry_question,
            })
            continue

        valides.append({
            "id_test": id_test,
            "categorie": golden_q["categorie"],
            "question": golden_q,
            "mots_cles_generes": mots_cles,
        })

    return valides, orphelins, incoherentes


# ── Étape 2-3 : requête augmentée + relance dense/hybride ───────────────────

def build_augmented_query(question_client: str, mots_cles: list[str]) -> str:
    """Même format exact que test_keyword_expansion.py::build_augmented_query."""
    return f"{question_client} {' '.join(mots_cles)}"


def rank_of(ground_truth_id: str, top_ids: list[str]) -> int | None:
    return top_ids.index(ground_truth_id) + 1 if ground_truth_id in top_ids else None


def evaluate_generated_query(
    valide_entry: dict[str, Any],
    model: SentenceTransformer,
    bm25,
    vocab: dict[str, int],
    client: QdrantClient,
    collection_name: str,
    limit: int = INSPECTION_LIMIT,
) -> dict[str, Any]:
    question = valide_entry["question"]
    ground_truth_id = build_point_id(question["id_reference"], question["source_type"])

    query_augmentee = build_augmented_query(question["question_client"], valide_entry["mots_cles_generes"])

    dense_query = model.encode(
        query_augmentee, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    sparse_query = vectorize_bm25(tokenize(query_augmentee), bm25, vocab)

    dense_top_ids = [pid for pid, _ in dense_only_search(client, collection_name, dense_query, limit=limit)]
    hybride_top_ids = [
        pid for pid, _ in hybrid_search(client, collection_name, dense_query, sparse_query, limit=limit)
    ]

    return {
        "id_test": valide_entry["id_test"],
        "categorie": valide_entry["categorie"],
        "query_augmentee": query_augmentee,
        "rang_genere_dense": rank_of(ground_truth_id, dense_top_ids),
        "rang_genere_hybride": rank_of(ground_truth_id, hybride_top_ids),
    }


# ── Étape 4 : récupération des rangs déjà connus (baseline/oracle) ──────────

def load_baseline_ranks(hybrid_report_path: str, dense_report_path: str) -> dict[str, dict[str, Any]]:
    """
    Récupère les rangs baseline (sans expansion) déjà calculés par
    evaluate_retrieval.py, sans relancer aucun calcul. Clé : id_test.
    """
    baseline: dict[str, dict[str, Any]] = {}

    if Path(hybrid_report_path).exists():
        with open(hybrid_report_path, "r", encoding="utf-8") as f:
            hybride = json.load(f)
        for d in hybride.get("details", []):
            if d.get("reference_invalide"):
                continue
            baseline.setdefault(d["id_test"], {})["rang_baseline_hybride"] = d.get("rang_trouve")

    if Path(dense_report_path).exists():
        with open(dense_report_path, "r", encoding="utf-8") as f:
            dense = json.load(f)
        for d in dense.get("details", []):
            if d.get("reference_invalide"):
                continue
            baseline.setdefault(d["id_test"], {})["rang_baseline_dense"] = d.get("rang_trouve")

    return baseline


def load_oracle_ranks_from_prior_run(oracle_data: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """
    Rangs oracle déjà mesurés par test_keyword_expansion.py (rapportés dans la
    conversation précédente, pas de fichier de rapport JSON généré à l'époque).
    Fournis ici en dur, ré-utilisés tels quels (pas recalculés) :
      - gd_facturation_006 (Devis)    : dense-only rang 7,  hybride rang 1
      - gd_facturation_011 (Forfaits) : dense-only rang 20, hybride rang 1
    """
    return oracle_data or {
        "gd_facturation_006": {"rang_oracle_dense": 7, "rang_oracle_hybride": 1},
        "gd_facturation_011": {"rang_oracle_dense": 20, "rang_oracle_hybride": 1},
    }


# ── Étape 6-7 : écarts et conclusion ─────────────────────────────────────────

def format_rang(rang: int | None) -> str:
    return str(rang) if rang is not None else f"absent du top-{INSPECTION_LIMIT}"


def compute_ecart(a: int | None, b: int | None) -> int | None:
    """Écart de rang (positif = a est pire/plus grand que b). None si l'un des deux est absent."""
    if a is None or b is None:
        return None
    return a - b


def conclure(rang_baseline: int | None, rang_oracle: int | None, rang_genere: int | None) -> str:
    """
    Conclusion PONCTUELLE pour cette question précise — pas de généralisation
    à toute une catégorie, cf. consigne explicite.
    """
    baseline_absent = rang_baseline is None
    genere_absent = rang_genere is None

    if genere_absent:
        return "insuffisant vs plafond (mots-clés générés : toujours absent, aucun gain mesurable)"

    if not baseline_absent and rang_genere >= rang_baseline:
        return "neutre (généré n'améliore pas, voire dégrade, par rapport à la baseline)"

    # généré apporte un gain par rapport à la baseline (ou baseline absente et généré trouvé)
    if rang_oracle is not None and rang_genere > rang_oracle + 2:
        return "gain net mais loin du plafond oracle (expansion utile mais mots-clés générés sous-optimaux)"
    if rang_oracle is not None and rang_genere <= rang_oracle + 2:
        return "gain net et proche du plafond oracle (expansion réaliste solide sur ce cas)"
    return "gain net (généré trouvé, pas de référence oracle pour comparer l'écart)"


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("COMPARAISON BASELINE / ORACLE / GÉNÉRÉ — expansion de requête par mots-clés")
    print("=" * 80)

    golden_questions = load_golden_dataset(DEFAULT_GOLDEN_PATH)

    print(f"\n[ÉTAPE 1] Chargement et validation de {GENERATED_KEYWORDS_PATH}")
    try:
        raw = load_generated_keywords_file(GENERATED_KEYWORDS_PATH)
        generated_entries = extract_generated_entries(raw)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"\n[ERREUR FATALE] {exc}")
        print("Arrêt du script : impossible de valider le fichier d'entrée. Aucun calcul effectué.")
        sys.exit(1)

    valides, orphelins, incoherentes = validate_entries_against_golden(generated_entries, golden_questions)

    if orphelins:
        print(f"\n[ERREUR] {len(orphelins)} id_test du fichier généré sont ABSENTS de golden_dataset.json :")
        for o in orphelins:
            print(f"    - {o!r}")

    if incoherentes:
        print(f"\n[ERREUR] {len(incoherentes)} entrée(s) avec un id_test connu mais un contenu incohérent :")
        for inc in incoherentes:
            print(f"    - {inc['id_test']} : {inc['reason']}")
            if "question_golden_dataset" in inc:
                print(f"        golden   : {inc['question_golden_dataset'][:100]}")
                print(f"        fichier  : {inc['question_fichier_genere'][:100]}")

    if orphelins or incoherentes:
        print(
            f"\n[ARRÊT] Le fichier {GENERATED_KEYWORDS_PATH} ne peut pas être utilisé tel quel : "
            f"{len(orphelins)} id_test orphelin(s), {len(incoherentes)} entrée(s) incohérente(s), "
            f"{len(valides)} entrée(s) exploitable(s) sur {len(generated_entries)}.\n"
            "Conformément à la consigne (« ne suppose rien, échoue explicitement, ne traite pas "
            "silencieusement »), le script s'arrête ici sans calculer de rangs sur des données non fiables."
        )
        report = {
            "statut": "echec_validation",
            "nb_entrees_fichier": len(generated_entries),
            "nb_valides": len(valides),
            "id_tests_orphelins": orphelins,
            "entrees_incoherentes": incoherentes,
        }
        Path(COMPARISON_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] État de l'échec de validation sauvegardé dans {COMPARISON_REPORT_PATH}")
        sys.exit(1)

    print(f"\n[OK] {len(valides)}/{len(generated_entries)} entrées valides, aucune incohérence détectée.")

    # ── Setup partagé pour le calcul dense/hybride ──────────────────────────
    ingestion_config = load_ingestion_config(DEFAULT_INGESTION_CONFIG_PATH)
    texts, _ = build_reference_corpus(DEFAULT_INGESTION_CONFIG_PATH)
    bm25, vocab = build_bm25_index(texts)
    model = SentenceTransformer(ingestion_config["modele_embedding"])
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, timeout=30)
    collection_name = ingestion_config["collection_name"]

    print(f"\n[ÉTAPE 2-3] Requête augmentée (mots-clés générés) + relance dense/hybride, profondeur {INSPECTION_LIMIT}")
    resultats_generes = [
        evaluate_generated_query(v, model, bm25, vocab, client, collection_name, limit=INSPECTION_LIMIT)
        for v in valides
    ]

    print(f"\n[ÉTAPE 4] Récupération des rangs baseline/oracle déjà connus (aucun recalcul)")
    baseline_par_id = load_baseline_ranks(HYBRID_REPORT_PATH, DENSE_ONLY_REPORT_PATH)
    oracle_par_id = load_oracle_ranks_from_prior_run(None)

    print("\n" + "=" * 80)
    print("[ÉTAPE 5] TABLEAU COMPARATIF")
    print("=" * 80)
    header = (
        f"{'id_test':<22} {'categorie':<15} "
        f"{'base_H':>7} {'orcl_H':>7} {'gen_H':>7}  "
        f"{'base_D':>7} {'orcl_D':>7} {'gen_D':>7}"
    )
    print(header)

    rapport_detail = []
    for r in resultats_generes:
        id_test = r["id_test"]
        base = baseline_par_id.get(id_test, {})
        oracle = oracle_par_id.get(id_test, {})

        rang_base_h = base.get("rang_baseline_hybride")
        rang_base_d = base.get("rang_baseline_dense")
        rang_oracle_h = oracle.get("rang_oracle_hybride")
        rang_oracle_d = oracle.get("rang_oracle_dense")
        rang_gen_h = r["rang_genere_hybride"]
        rang_gen_d = r["rang_genere_dense"]

        print(
            f"{id_test:<22} {r['categorie']:<15} "
            f"{format_rang(rang_base_h):>7} {format_rang(rang_oracle_h):>7} {format_rang(rang_gen_h):>7}  "
            f"{format_rang(rang_base_d):>7} {format_rang(rang_oracle_d):>7} {format_rang(rang_gen_d):>7}"
        )

        ecart_oracle_genere_h = compute_ecart(rang_gen_h, rang_oracle_h)
        ecart_baseline_genere_h = compute_ecart(rang_base_h, rang_gen_h)
        conclusion_h = conclure(rang_base_h, rang_oracle_h, rang_gen_h)

        rapport_detail.append({
            "id_test": id_test,
            "categorie": r["categorie"],
            "query_augmentee": r["query_augmentee"],
            "hybride": {
                "rang_baseline": rang_base_h,
                "rang_oracle": rang_oracle_h,
                "rang_genere": rang_gen_h,
                "ecart_oracle_vs_genere": ecart_oracle_genere_h,
                "ecart_baseline_vs_genere": ecart_baseline_genere_h,
                "conclusion": conclusion_h,
            },
            "dense_only": {
                "rang_baseline": rang_base_d,
                "rang_oracle": rang_oracle_d,
                "rang_genere": rang_gen_d,
                "ecart_oracle_vs_genere": compute_ecart(rang_gen_d, rang_oracle_d),
                "ecart_baseline_vs_genere": compute_ecart(rang_base_d, rang_gen_d),
                "conclusion": conclure(rang_base_d, rang_oracle_d, rang_gen_d),
            },
        })

    print("\n" + "=" * 80)
    print("[ÉTAPE 6-7] ÉCARTS ET CONCLUSIONS PONCTUELLES (par id_test, hybride)")
    print("=" * 80)
    for d in rapport_detail:
        h = d["hybride"]
        print(f"\n  {d['id_test']} ({d['categorie']})")
        print(f"    rang baseline={format_rang(h['rang_baseline'])} | oracle={format_rang(h['rang_oracle'])} | genere={format_rang(h['rang_genere'])}")
        print(f"    ecart oracle-genere={h['ecart_oracle_vs_genere']} | ecart baseline-genere={h['ecart_baseline_vs_genere']}")
        print(f"    conclusion : {h['conclusion']}")

    report = {
        "statut": "ok",
        "profondeur_inspection": INSPECTION_LIMIT,
        "nb_questions_comparees": len(rapport_detail),
        "details": rapport_detail,
    }
    Path(COMPARISON_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Rapport sauvegardé dans {COMPARISON_REPORT_PATH}")


if __name__ == "__main__":
    main()
