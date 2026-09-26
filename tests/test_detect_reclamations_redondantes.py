"""
tests/test_detect_reclamations_redondantes.py
Tests de data_pipeline/detect_reclamations_redondantes.py :
  0. chargement + validation des colonnes
  1. bucketing (module + mots-clés) -> sous_theme / bucket_id
  2. fusion des doublons exacts (hash_doublon) intra-bucket
  3. embeddings : construction texte + nettoyage du bruit
  4. similarité cosinus + composantes connexes (union-find)
  5-6. orchestration dry-run / sortie finale (encodeur injecté)

Un test d'intégration (marqué) charge le vrai modèle sentence-transformers
pour mesurer l'effet du nettoyage de bruit sur des paires réelles.
"""

import numpy as np
import pandas as pd
import pytest

from data_pipeline.detect_reclamations_redondantes import (
    assemble_clusters,
    assign_buckets,
    build_bucket_report,
    build_dryrun_pairs,
    build_embedding_text,
    cluster_indices,
    connected_components,
    cosine_matrix,
    count_informative_tokens,
    fuse_exact_duplicates,
    load_keywords,
    load_reclamations,
    prepare_representatives,
    run_redundancy_detection,
    _strip_embedding_noise,
    DEFAULT_INPUT,
    DEFAULT_KEYWORDS,
    NON_CATEGORISE,
)


# ── Fixtures : dictionnaire de mots-clés synthétique ──────────────────────────
# Volontairement séparé du vrai module_keywords.json (encore "a_valider") pour
# des tests déterministes. Ordre des sous-thèmes = ordre d'insertion (préservé
# par dict Python 3.7+ et par json.load).

@pytest.fixture
def kw_config():
    return {
        "modules": {
            "Financier": {
                "rapprochement_bancaire": {
                    "keywords": ["rapprochement", "trésorerie"],
                    "variantes": ["injection rapprochement"],
                },
                "gestion_bordereaux": {
                    "keywords": ["bordereau", "versement"],
                    "variantes": [],
                },
            },
            # Clé JSON SANS accent, alors que le CSV porte "Comptabilité".
            "Comptabilite": {
                "flux_integration": {
                    "keywords": ["intégration comptable"],
                    "variantes": ["intégration vente"],
                },
                "declarations": {
                    # mot-clé court (<= 4 car.) -> matching par limite de mot
                    "keywords": ["TVA", "REG"],
                    "variantes": [],
                },
            },
        }
    }


def _row(idm, module, texte, hash_doublon="h", objet="obj", message="msg", idclient=1):
    return {
        "idMessage": idm,
        "module_designation": module,
        "objet_clean": objet,
        "message_clean": message,
        "texte_complet": texte,
        "hash_doublon": hash_doublon,
        "IdClient": idclient,
    }


# ── Phase 0 : chargement / validation ─────────────────────────────────────────

def test_load_reclamations_reads_semicolon_and_comma(tmp_path):
    path = tmp_path / "clean.csv"
    df_in = pd.DataFrame([
        _row(1, "Financier", "rapprochement du module"),
        _row(2, "Facturation", "changement de nom"),
    ])
    df_in.to_csv(path, index=False, encoding="utf-8-sig")  # séparateur ","
    df = load_reclamations(path)
    assert len(df) == 2
    assert set(["idMessage", "module_designation", "texte_complet", "hash_doublon"]).issubset(df.columns)


def test_load_reclamations_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"idMessage": 1, "module_designation": "Financier"}]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    with pytest.raises(Exception):
        load_reclamations(path)


# ── Phase 1 : bucketing ───────────────────────────────────────────────────────

def test_bucketing_assigns_correct_sous_theme(kw_config):
    df = pd.DataFrame([_row(1, "Financier", "problème de rapprochement au niveau du module")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "rapprochement_bancaire"


def test_bucketing_no_keyword_gives_non_categorise(kw_config):
    df = pd.DataFrame([_row(1, "Financier", "le bouton ne fonctionne pas du tout ici")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == NON_CATEGORISE


def test_bucketing_module_match_is_accent_and_case_insensitive(kw_config):
    # module CSV "Comptabilité" (accent) doit matcher la clé JSON "Comptabilite"
    # et le mot-clé accentué doit matcher un texte en majuscules sans accent.
    df = pd.DataFrame([_row(1, "Comptabilité", "INTEGRATION COMPTABLE de la paie")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "flux_integration"


def test_bucketing_module_with_trailing_spaces_still_matches(kw_config):
    # Cas réel : les module_designation du CSV portent des espaces en fin
    # ("Facturation  ", "Comptabilité ") issus de l'export SQL. Le matching de
    # module doit rester robuste (sinon 100 % des tickets tombent en
    # non_categorise — bug rencontré sur données réelles).
    df = pd.DataFrame([
        _row(1, "Comptabilité ", "problème d'intégration comptable de la paie"),
        _row(2, "Financier  ", "problème de rapprochement bancaire"),
    ])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "flux_integration"
    assert out.iloc[1]["sous_theme"] == "rapprochement_bancaire"


def test_bucketing_variante_counts_as_keyword(kw_config):
    df = pd.DataFrame([_row(1, "Comptabilite", "problème d'intégration vente ce mois")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "flux_integration"


def test_bucketing_short_keyword_uses_word_boundary(kw_config):
    # "REG" (<=4 car.) ne doit PAS matcher à l'intérieur de "reglement"
    df = pd.DataFrame([_row(1, "Comptabilite", "le reglement fournisseur du dossier")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == NON_CATEGORISE

    # mais DOIT matcher "REG" isolé
    df2 = pd.DataFrame([_row(2, "Comptabilite", "voir le code REG dans la pièce")])
    out2, _ = assign_buckets(df2, kw_config)
    assert out2.iloc[0]["sous_theme"] == "declarations"


def test_bucketing_short_keyword_tva_word_boundary(kw_config):
    # "TVA" isolé matche ; à l'intérieur d'un mot (fabriqué) ne matche pas.
    df = pd.DataFrame([
        _row(1, "Comptabilite", "la TVA est mal calculée"),
        _row(2, "Comptabilite", "aucun terme metztvashi pertinent ici présent"),
    ])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "declarations"
    assert out.iloc[1]["sous_theme"] == NON_CATEGORISE


def test_bucketing_tiebreak_by_match_count(kw_config):
    # 1 mot-clé de rapprochement_bancaire vs 2 de gestion_bordereaux -> bordereaux
    df = pd.DataFrame([_row(1, "Financier", "rapprochement du bordereau de versement")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "gestion_bordereaux"


def test_bucketing_tiebreak_by_json_order_on_equal_count(kw_config):
    # 1 match dans chaque sous-thème (égalité) -> premier dans l'ordre JSON
    # (rapprochement_bancaire précède gestion_bordereaux).
    df = pd.DataFrame([_row(1, "Financier", "un rapprochement et un bordereau")])
    out, _ = assign_buckets(df, kw_config)
    assert out.iloc[0]["sous_theme"] == "rapprochement_bancaire"


def test_bucketing_report_counts_multi_subtheme_tickets(kw_config):
    df = pd.DataFrame([
        _row(1, "Financier", "un rapprochement et un bordereau"),   # 2 sous-thèmes
        _row(2, "Financier", "juste un rapprochement simple ici"),  # 1 sous-thème
    ])
    out, report = assign_buckets(df, kw_config)
    assert report["n_tickets_multi_sous_theme"] == 1


# ── Phase 2 : fusion des doublons exacts ──────────────────────────────────────

def test_exact_fusion_same_bucket_same_hash_merges(kw_config):
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement version courte", hash_doublon="AAA"),
        _row(2, "Financier", "rapprochement version beaucoup plus longue ici", hash_doublon="AAA"),
    ])
    bucketed, _ = assign_buckets(df, kw_config)
    fused = fuse_exact_duplicates(bucketed)
    # même bucket + même hash -> un seul groupe de 2, un seul représentant
    assert fused["exact_group_id"].nunique() == 1
    assert fused["is_representant_exact"].sum() == 1
    # représentant = texte le plus long
    rep = fused[fused["is_representant_exact"]].iloc[0]
    assert rep["idMessage"] == 2


def test_exact_fusion_same_hash_different_bucket_not_merged(kw_config):
    # même hash mais buckets différents (rapprochement vs bordereau) -> 2 groupes
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement trésorerie", hash_doublon="SAME"),
        _row(2, "Financier", "bordereau de versement", hash_doublon="SAME"),
    ])
    bucketed, _ = assign_buckets(df, kw_config)
    assert bucketed["sous_theme"].nunique() == 2  # pré-condition : buckets distincts
    fused = fuse_exact_duplicates(bucketed)
    assert fused["exact_group_id"].nunique() == 2
    assert fused["is_representant_exact"].sum() == 2


def test_exact_fusion_representative_is_longest_text(kw_config):
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement court", hash_doublon="H"),
        _row(2, "Financier", "rapprochement moyen ici", hash_doublon="H"),
        _row(3, "Financier", "rapprochement le plus long de tous les trois", hash_doublon="H"),
    ])
    bucketed, _ = assign_buckets(df, kw_config)
    fused = fuse_exact_duplicates(bucketed)
    rep = fused[fused["is_representant_exact"]].iloc[0]
    assert rep["idMessage"] == 3
    assert int(fused["exact_group_size"].iloc[0]) == 3


def test_exact_fusion_unique_hashes_each_singleton(kw_config):
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement A", hash_doublon="H1"),
        _row(2, "Financier", "rapprochement B", hash_doublon="H2"),
    ])
    bucketed, _ = assign_buckets(df, kw_config)
    fused = fuse_exact_duplicates(bucketed)
    assert fused["exact_group_id"].nunique() == 2
    assert fused["is_representant_exact"].sum() == 2  # chacun son propre représentant


# ── Dictionnaire réel v0.2 : nouveau sous-thème immobilisations ───────────────

def test_real_dict_buckets_immobilisations_ticket_1580():
    # Sous-thème "immobilisations" ajouté en v0.2. Cas réel du corpus
    # doit désormais matcher (avant v0.2 -> non_categorise).
    # On teste contre le VRAI module_keywords.json, pas la fixture synthétique.
    kw = load_keywords(DEFAULT_KEYWORDS)
    df = pd.DataFrame([_row(
        1580,
        "Comptabilité",
        "Liste des Immobilisations 100% Ci-joint un fichier contenant "
        "liste des immobilisations avec taux 100%",
    )])
    out, _ = assign_buckets(df, kw)
    assert out.iloc[0]["sous_theme"] == "immobilisations"


def test_real_dict_amortissement_matches_immobilisations():
    # Le mot-clé "amortissement" (v0.2) rattache aussi à immobilisations.
    kw = load_keywords(DEFAULT_KEYWORDS)
    df = pd.DataFrame([_row(
        9001,
        "Comptabilité",
        "problème de calcul d'amortissement sur le dernier exercice",
    )])
    out, _ = assign_buckets(df, kw)
    assert out.iloc[0]["sous_theme"] == "immobilisations"


# ── build_bucket_report ───────────────────────────────────────────────────────

def test_build_bucket_report_counts_per_bucket(kw_config):
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement trésorerie"),
        _row(2, "Financier", "bordereau de versement"),
        _row(3, "Financier", "le bouton ne marche pas ici du tout"),  # non_categorise
    ])
    bucketed, _ = assign_buckets(df, kw_config)
    report = build_bucket_report(bucketed)
    counts = report["counts_par_bucket"]
    # 3 buckets distincts, 1 ticket chacun
    assert counts[("Financier", "rapprochement_bancaire")] == 1
    assert counts[("Financier", "gestion_bordereaux")] == 1
    assert counts[("Financier", NON_CATEGORISE)] == 1


# ── Phase 3 : build_embedding_text ────────────────────────────────────────────

def test_build_embedding_text_repeats_subject_and_truncates():
    objet = "assistance module financier"
    message = "x" * 500
    out = build_embedding_text(objet, message, truncate=400)
    # sujet répété deux fois, puis message tronqué à 400
    assert out == f"{objet}. {objet}. " + "x" * 400
    assert out.count(objet) == 2
    assert ("x" * 400) in out
    assert ("x" * 401) not in out


def test_build_embedding_text_empty_objet_fallback():
    out = build_embedding_text("", "y" * 500, truncate=400)
    # pas de préfixe ". . " parasite, juste le message tronqué
    assert out == "y" * 400
    assert not out.startswith(".")


def test_build_embedding_text_handles_non_string():
    assert build_embedding_text(None, None, truncate=400) == ""


# ── Phase 3 : nettoyage du bruit d'embedding (_strip_embedding_noise) ──────────

def test_strip_noise_removes_greetings_and_collaborator_names():
    out = _strip_embedding_noise("BJR COLLABORATEUR voila le probleme integration paie merci")
    low = out.lower()
    for noise in ("bjr", "collaborateur", "voila", "probleme", "merci"):
        assert noise not in low
    assert "integration" in low
    assert "paie" in low


def test_strip_noise_handles_punctuation_glued_greetings():
    # "BJR.VOILA LE PROBLEME" (pas d'espace après BJR) : entièrement du bruit
    # -> chaîne vide après nettoyage (y compris le "." résiduel).
    assert _strip_embedding_noise("BJR.VOILA LE PROBLEME") == ""
    assert _strip_embedding_noise("BJR .VOILA LE PB MERCI") == ""


def test_strip_noise_keeps_informative_content_untouched():
    # Texte métier sans politesse : rien ne doit être retiré.
    text = "balance d'ouverture non equilibree au niveau du module comptabilite"
    assert _strip_embedding_noise(text) == text


def test_strip_noise_keeps_content_removes_only_greeting():
    out = _strip_embedding_noise(
        "bonjour Collaborateur Alpha, Je vous rappelle l'envoi de la balance d'ouverture 2014"
    )
    assert "bonjour" not in out.lower()
    assert "collaborateur" not in out.lower()
    assert "balance d'ouverture" in out
    assert "2014" in out


def test_build_embedding_text_strips_noise_short_ticket():
    # Objet informatif, message = pure politesse -> repli sur
    # le sujet répété seul, sans bruit.
    out = build_embedding_text("INTEGRATION CA", "BJR.VOILA LE PROBLEME", truncate=400)
    assert "INTEGRATION CA" in out
    assert "BJR" not in out.upper()
    assert "PROBLEME" not in out.upper()


def test_build_embedding_text_fallback_when_objet_all_noise():
    # objet entièrement bruit -> repli sur message nettoyé, pas de "INTEGRATION"
    # dupliqué ni préfixe ". . ".
    out = build_embedding_text("BJR", "integration facture economat", truncate=400)
    assert out == "integration facture economat"


# ── Phase 4 : cosinus + union-find ────────────────────────────────────────────

def test_cosine_matrix_identical_and_orthogonal():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sim = cosine_matrix(v)
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(0.0)


def test_connected_components_basic_and_chaining():
    # 5 nœuds, arêtes 0-1, 1-2 (chaînage), 3-4 ; 0-2 absent mais chaînés via 1
    comps = connected_components(5, [(0, 1), (1, 2), (3, 4)])
    comps_sets = sorted([sorted(c) for c in comps])
    assert comps_sets == [[0, 1, 2], [3, 4]]


def test_cluster_indices_threshold_is_configurable():
    # v0~v1 = 0.9 ; v0~v2 = 0 ; v1~v2 = 0.436
    v = np.array([[1.0, 0.0], [0.9, np.sqrt(1 - 0.81)], [0.0, 1.0]])
    # seuil 0.85 : seule l'arête 0-1 -> {0,1}, {2}
    comps = sorted([sorted(c) for c in cluster_indices(v, threshold=0.85)])
    assert comps == [[0, 1], [2]]
    # seuil 0.40 : 0-1 (0.9) et 1-2 (0.436) -> chaînage {0,1,2}
    comps_low = sorted([sorted(c) for c in cluster_indices(v, threshold=0.40)])
    assert comps_low == [[0, 1, 2]]


def test_cluster_indices_singleton_and_empty():
    assert cluster_indices(np.zeros((0, 3)), 0.85) == []
    assert cluster_indices(np.array([[1.0, 0.0]]), 0.85) == [[0]]


# ── Phases 5-6 : orchestration avec encodeur injecté ──────────────────────────

def _fake_encoder(vec_by_text):
    """Retourne un encodeur qui mappe chaque texte à un vecteur fixe (2D unit)."""
    def _encode(texts, model_name):
        return np.array([vec_by_text[t] for t in texts], dtype=float)
    return _encode


def test_dry_run_writes_pairs_not_final_output(tmp_path, kw_config):
    # 2 tickets Financier même bucket (rapprochement), textes distincts mais
    # vecteurs proches -> une paire au-dessus du plancher.
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement A", hash_doublon="H1", objet="rapprochement A", message="souci de rapprochement bancaire un"),
        _row(2, "Financier", "rapprochement B", hash_doublon="H2", objet="rapprochement B", message="souci de rapprochement bancaire deux"),
    ])
    reps = prepare_representatives(fuse_exact_duplicates(assign_buckets(df, kw_config)[0]), truncate=400, kw_config=kw_config, min_informative_tokens=0)
    texts = list(reps["texte_pour_embedding"])
    vec_by_text = {texts[0]: [1.0, 0.0], texts[1]: [0.99, np.sqrt(1 - 0.99 ** 2)]}

    result = run_redundancy_detection(
        df, kw_config, threshold=0.85, dryrun_floor=0.70, truncate=400,
        model_name="dummy", dry_run=True, output_dir=tmp_path,
        encode_fn=_fake_encoder(vec_by_text),
    )
    # dry-run : rapport de paires écrit, sortie finale ABSENTE
    assert (tmp_path / "reclamations_redondantes_dryrun.csv").exists()
    assert not (tmp_path / "reclamations_redondantes.csv").exists()
    assert not (tmp_path / "reclamations_redondantes_membres.csv").exists()
    assert len(result["dryrun_pairs"]) == 1
    assert result["dryrun_pairs"].iloc[0]["cosinus"] >= 0.70


def test_final_output_columns_and_methods(tmp_path, kw_config):
    # bucket rapprochement : 2 tickets sémantiquement proches (cos 0.99) + 1
    # doublon exact ; bucket bordereau : 1 singleton.
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement tresorerie un", hash_doublon="A", objet="rapprochement", message="tresorerie un", idclient=10),
        _row(2, "Financier", "rapprochement tresorerie deux", hash_doublon="B", objet="rapprochement", message="tresorerie deux", idclient=20),
        _row(3, "Financier", "rapprochement tresorerie un", hash_doublon="A", objet="rapprochement", message="tresorerie un", idclient=10),  # doublon exact de 1
        _row(4, "Financier", "bordereau de versement seul", hash_doublon="C", objet="bordereau versement", message="seul", idclient=30),
    ])
    reps = prepare_representatives(fuse_exact_duplicates(assign_buckets(df, kw_config)[0]), truncate=400, kw_config=kw_config, min_informative_tokens=0)
    texts = list(reps["texte_pour_embedding"])
    # représentants : ticket 1 (groupe exact A, texte identique 1&3), ticket 2, ticket 4
    vecs = {}
    for t in texts:
        if "tresorerie un" in t:
            vecs[t] = [1.0, 0.0]
        elif "tresorerie deux" in t:
            vecs[t] = [0.99, np.sqrt(1 - 0.99 ** 2)]
        else:  # bordereau
            vecs[t] = [0.0, 1.0]

    result = run_redundancy_detection(
        df, kw_config, threshold=0.85, dryrun_floor=0.70, truncate=400,
        model_name="dummy", dry_run=False, output_dir=tmp_path,
        encode_fn=_fake_encoder(vecs),
    )
    clusters = result["clusters"]
    membres = result["membres"]

    expected_cols = {
        "cluster_id", "module", "sous_theme", "methode_fusion", "nb_tickets",
        "nb_clients_distincts", "idMessages_membres", "idMessage_representant",
        "score_similarite_min", "texte_representant",
    }
    assert expected_cols.issubset(set(clusters.columns))

    # rapprochement : cluster sémantique (tickets 1+3 exact fusionnés, + 2) = 3 tickets
    rappr = clusters[clusters["sous_theme"] == "rapprochement_bancaire"]
    assert len(rappr) == 1
    row = rappr.iloc[0]
    assert row["methode_fusion"] == "semantique"
    assert row["nb_tickets"] == 3
    # clients distincts : 10 (x2) et 20 -> 2
    assert row["nb_clients_distincts"] == 2

    # bordereau : singleton
    bord = clusters[clusters["sous_theme"] == "gestion_bordereaux"].iloc[0]
    assert bord["methode_fusion"] == "singleton"
    assert bord["nb_tickets"] == 1

    # membres : une ligne par ticket (4 tickets)
    assert len(membres) == 4
    assert set(membres["idMessage"]) == {1, 2, 3, 4}


def test_final_output_exact_method_when_only_duplicates(tmp_path, kw_config):
    # 2 doublons exacts, aucun voisin sémantique -> methode "exact"
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement identique", hash_doublon="Z", objet="rapprochement", message="identique", idclient=5),
        _row(2, "Financier", "rapprochement identique", hash_doublon="Z", objet="rapprochement", message="identique", idclient=6),
    ])
    reps = prepare_representatives(fuse_exact_duplicates(assign_buckets(df, kw_config)[0]), truncate=400, kw_config=kw_config, min_informative_tokens=0)
    texts = list(reps["texte_pour_embedding"])
    vecs = {t: [1.0, 0.0] for t in texts}
    result = run_redundancy_detection(
        df, kw_config, threshold=0.85, dryrun_floor=0.70, truncate=400,
        model_name="dummy", dry_run=False, output_dir=tmp_path,
        encode_fn=_fake_encoder(vecs),
    )
    clusters = result["clusters"]
    assert len(clusters) == 1
    assert clusters.iloc[0]["methode_fusion"] == "exact"
    assert clusters.iloc[0]["nb_tickets"] == 2
    assert clusters.iloc[0]["nb_clients_distincts"] == 2


# ── Intégration : effet du nettoyage de bruit sur le cosinus (vrai modèle) ────
# Charge le vrai modèle sentence-transformers (lent, ~10-15s). Vérifie que le
# nettoyage FAIT BAISSER la similarité des paires bruitées (7316/10697/572) et
# NE DÉGRADE PAS la paire à vrai contenu métier (3742/3843).

# Textes réels extraits de reclamations_seules.csv (objet_clean, message_clean)
_REAL = {
    7316: ("INTEGRATION CA", "BJR.VOILA LE PROBLEME"),
    10697: ("INTEGRATION FAUSSE", "BJR .VOILA LE PB MERCI"),
    572: ("INTEGRATION", "LE MODULE INTEGRATION NE FONCTIONNE PAS"),
    3742: ("balance d'ouverture",
           "bonjour Collaborateur Alpha, Je vous rappelle que vous n'avez pas encore envoyer "
           "la balance d'ouverture de 2014 avec le détails des libellés que je vous "
           "ai demandé lors de votre dernier passage"),
    3843: ("balance d'ouverture",
           "Bonjour, Je vous prie de bien vouloir trouver ci-joint le message "
           "d'erreur lors de l'envoi de la balance d'ouverture de 2014 . Merci de "
           "faire le nécessaire dans les plus brefs délais"),
}

_TRUNC = 400


def _raw_embedding_text(objet, message, truncate=_TRUNC):
    """Ancienne formule SANS nettoyage de bruit (pour comparaison)."""
    return f"{objet}. {objet}. {message[:truncate]}"


def test_noise_cleaning_effect_on_real_pairs_real_model():
    """
    Effet du nettoyage de bruit sur des paires réelles (mesuré sur le vrai
    modèle). Constats validés empiriquement :
      - 7316/10697 (faux positif à résidu différenciant "CA" vs "FAUSSE") :
        la similarité CHUTE nettement et repasse SOUS le seuil 0.85.
      - 3742/3843 (vrai doublon "balance d'ouverture") : NON dégradé (renforcé,
        car les greetings différents qui les éloignaient sont retirés).
    Le cas 572/10697 n'est volontairement PAS asserté ici : 572 est un ticket à
    très faible information (contenu réel = "intégration" répété), sa similarité
    résiduelle ne relève pas du bruit de politesse — limite documentée, pas un
    échec du nettoyage.
    """
    st = pytest.importorskip("sentence_transformers")
    from ai_engine.config import get_settings

    model = st.SentenceTransformer(get_settings().embedding_model)

    def raw_cos(id1, id2):
        v = model.encode([_raw_embedding_text(*_REAL[id1]), _raw_embedding_text(*_REAL[id2])],
                         normalize_embeddings=True, convert_to_numpy=True)
        return float(v[0] @ v[1])

    def clean_cos(id1, id2):
        v = model.encode([build_embedding_text(*_REAL[id1], truncate=_TRUNC),
                          build_embedding_text(*_REAL[id2], truncate=_TRUNC)],
                         normalize_embeddings=True, convert_to_numpy=True)
        return float(v[0] @ v[1])

    # faux positif clair : chute nette + repasse sous le seuil de regroupement
    before, after = raw_cos(7316, 10697), clean_cos(7316, 10697)
    assert after < before - 0.10, f"7316/10697 attendu en forte baisse (avant={before:.3f}, après={after:.3f})"
    assert after < 0.85, f"7316/10697 devrait repasser sous 0.85 (après={after:.3f})"

    # vrai doublon métier : non dégradé par le nettoyage
    before_good, after_good = raw_cos(3742, 3843), clean_cos(3742, 3843)
    assert after_good >= before_good - 0.03, (
        f"3742/3843 : contenu métier dégradé (avant={before_good:.3f}, après={after_good:.3f})"
    )


# ── Garde-fou "information minimale" (N=4, sur le VRAI corpus + dictionnaire) ──
# Les 9 idMessage suivants ont été identifiés par inspection manuelle comme
# génuinement vides de contenu différenciant (ne reste que le mot du bucket
# après nettoyage) : ils ne doivent JAMAIS être comparés entre eux ni à
# d'autres représentants. À l'inverse, 6982/6992/1580/25392 sont des cas
# limites mais informatifs (cf. analyse de fréquence) : ils doivent rester
# dans le pool comparable (is_low_info == False).
_KNOWN_EMPTY_IDS = {16815, 16796, 9751, 7899, 7316, 6422, 976, 19551, 10697}
_KNOWN_INFORMATIVE_BORDERLINE_IDS = {6982, 6992, 1580, 25392}
_MIN_TOKENS = 4


def _real_reps():
    df = load_reclamations(DEFAULT_INPUT)
    kw = load_keywords(DEFAULT_KEYWORDS)
    bucketed, _ = assign_buckets(df, kw)
    fused = fuse_exact_duplicates(bucketed)
    return prepare_representatives(fused, truncate=400, kw_config=kw,
                                   min_informative_tokens=_MIN_TOKENS)


def test_known_empty_tickets_are_flagged_low_info_on_real_corpus():
    reps = _real_reps()
    flagged = set(reps[reps["is_low_info"]]["idMessage"])
    missing = _KNOWN_EMPTY_IDS - flagged
    assert not missing, f"tickets attendus vides mais non flagués low-info : {missing}"


def test_known_borderline_informative_tickets_stay_comparable_on_real_corpus():
    reps = _real_reps()
    low_info = set(reps[reps["is_low_info"]]["idMessage"])
    wrongly_excluded = _KNOWN_INFORMATIVE_BORDERLINE_IDS & low_info
    assert not wrongly_excluded, (
        f"tickets informatifs exclus à tort par le garde-fou : {wrongly_excluded}"
    )


def test_low_info_reps_never_get_semantic_edges(kw_config):
    # 3 tickets d'un même bucket, textes tous vides après nettoyage (bruit only)
    # -> même avec des vecteurs identiques (cos=1.0), aucun ne doit être fusionné
    # sémantiquement avec un autre : is_low_info empêche toute comparaison.
    df = pd.DataFrame([
        _row(1, "Financier", "rapprochement A", objet="RAPPROCHEMENT", message="bjr merci", hash_doublon="H1"),
        _row(2, "Financier", "rapprochement B", objet="RAPPROCHEMENT", message="salut merci", hash_doublon="H2"),
        _row(3, "Financier", "rapprochement C", objet="RAPPROCHEMENT", message="bjr voila", hash_doublon="H3"),
    ])
    reps = prepare_representatives(
        fuse_exact_duplicates(assign_buckets(df, kw_config)[0]),
        truncate=400, kw_config=kw_config, min_informative_tokens=4,
    )
    assert reps["is_low_info"].all()  # "RAPPROCHEMENT" seul = terme du bucket, 0 token informatif

    vecs = np.array([[1.0, 0.0]] * len(reps))  # tous identiques : cos=1.0 partout
    clusters_df, membres_df = assemble_clusters(fuse_exact_duplicates(assign_buckets(df, kw_config)[0]), reps, vecs, threshold=0.85)

    assert len(clusters_df) == 3  # 3 singletons, jamais fusionnés
    assert set(clusters_df["methode_fusion"]) == {"singleton_info_faible"}
    assert set(clusters_df["nb_tickets"]) == {1}


def test_low_info_guard_disabled_when_min_tokens_is_zero(kw_config):
    # min_informative_tokens=0 -> aucun ticket ne peut avoir n_tokens < 0 :
    # comportement identique à avant le garde-fou (aucun singleton_info_faible).
    df = pd.DataFrame([_row(1, "Financier", "rapprochement A", objet="RAPPROCHEMENT", message="bjr merci")])
    reps = prepare_representatives(
        fuse_exact_duplicates(assign_buckets(df, kw_config)[0]),
        truncate=400, kw_config=kw_config, min_informative_tokens=0,
    )
    assert not reps["is_low_info"].any()


def test_count_informative_tokens_excludes_bucket_terms_as_whole_phrases():
    # Piège déjà rencontré : ne pas décomposer un terme multi-mots en mots
    # isolés. "paramétrage état résultat" ne doit pas retirer "paramétrage"
    # d'un texte qui parle de "paramétrage du bilan" (contenu différent).
    terms = ["paramétrage état résultat", "bilan"]
    text = "Assistance dans le paramétrage du bilan 2013"
    n = count_informative_tokens(text, terms)
    # "bilan" retiré (terme exact), mais "paramétrage" reste : au moins
    # assistance + paramétrage + 2013 comptent (2013 n'est pas alphabétique
    # -> non compté par le regex [a-zà-ÿ]{2,}, mais assistance/paramétrage oui)
    assert n >= 2
