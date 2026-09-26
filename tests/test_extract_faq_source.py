"""
tests/test_extract_faq_source.py
Tests du pipeline FAQ (data_pipeline/extract_faq_source.py) sur des exemples
réalistes : HTML Outlook/Word (commentaires conditionnels MSO) + image base64.
"""

import pandas as pd

from data_pipeline.extract_faq_source import normalize_category, process

# Exemple 1 : ticket Outlook typique (MSO + <o:p>) — doit être nettoyé et retenu
TICKET_OUTLOOK_FACTURATION = {
    "idMessage": 1001,
    "objet": "Erreur facture",
    "message": (
        "<!--[if gte mso 9]><xml><o:OfficeDocumentSettings>"
        "<o:AllowPNG/></o:OfficeDocumentSettings></xml><![endif]-->"
        "<p class=MsoNormal>Bonjour,<o:p></o:p></p>"
        "<p class=MsoNormal>La facture n&deg;2024-458 affiche un montant "
        "incorrect depuis la derni&egrave;re mise &agrave; jour du module "
        "facturation. Merci de corriger rapidement.<o:p></o:p></p>"
        "<p class=MsoNormal>Cordialement,<o:p></o:p></p>"
        "<p class=MsoNormal>Amal Trabelsi<o:p></o:p></p>"
    ),
    "module_designation": "Facturation  ",
    "objectif_reclamation": "Assistance",
    "IdClient": 42,
}

# Exemple 2 : ticket avec capture d'écran collée en base64 — doit être nettoyé
# (image neutralisée) mais retenu si le texte restant est suffisant.
TICKET_WITH_BASE64_IMAGE = {
    "idMessage": 1002,
    "objet": "Rapprochement bancaire bloqué",
    "message": (
        "<p>Voici la capture d'&eacute;cran du blocage lors du rapprochement "
        "bancaire, l'&eacute;cart de tr&eacute;sorerie persiste malgr&eacute; "
        "plusieurs tentatives.</p>"
        "<img src=\"data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQIHWNwUFH4z"
        "8DAwMDEwMDwn2GgAADgBQiA/2Q\">"
        "<p>Merci de v&eacute;rifier.</p>"
    ),
    "module_designation": "Financier",
    "objectif_reclamation": "Assistance",
    "IdClient": 43,
}

# Exemple 3 : ticket hors périmètre module (Support technique) — doit être rejeté
TICKET_OUT_OF_SCOPE_MODULE = {
    "idMessage": 1003,
    "objet": "Question générale",
    "message": "<p>Ceci est une question qui ne concerne pas un module financier.</p>",
    "module_designation": "Support technique",
    "objectif_reclamation": "Assistance",
    "IdClient": 44,
}

# Exemple 4 : texte trop court après nettoyage — doit être rejeté
TICKET_TOO_SHORT = {
    "idMessage": 1004,
    "objet": "Bug",
    "message": "<p>ok</p>",
    "module_designation": "Comptabilité",
    "objectif_reclamation": "Formation",
}

# ── Fixtures synthétiques de non-régression ───────────────────────────────────
# Utilisées pour Bug 1 (écriture CSV) et Bug 2 (généricisation des numéros métier).

# Cas de référence : objet avec virgule + suffixe collaborateur, message avec
# plusieurs références bordereau/chèque et montants.
TICKET_3926 = {
    "idMessage": 1001,
    "objet": "problème de rapprochement au niveau du module financier , à  collaborateur test",
    "message": (
        "bonjour<br>vous trouvez ci après les bordereaux de versement chèques que&nbsp; "
        "j'ai pas trouver dans la trésorerie pour les rapprocher:<br><br>"
        "bord n° 140005647 du 16/07/2014 : 588 .600 dt<br>"
        "bord n° 140005689 du 17/07/2014 : 494.317 dt<br>"
        "virement 4 228.760 dt du 24/07/2014 remplacement du chèque n° 1398594 "
        "(24 154.000DT ) ce virement a été intégré dans le compte b01; la banque correcte b05"
        "<br><br>merci pour votre intervention<br><br><br>Cordialement <br>Contact Test<br>"
    ),
    "module_designation": "Financier",
    "objectif_reclamation": "Assistance",
}

# Cas de référence : objet avec suffixe collaborateur (+ préfixe "*"), message
# avec "chèque n° ..." et montants à génériciser.
TICKET_4116 = {
    "idMessage": 1002,
    "objet": "*problème d'intégration des avances médecins , à  collaborateur test",
    "message": (
        '<p class="MsoNormal"><br></p><p class="MsoNormal">Bonjour</p>'
        '<p class="MsoNormal">Vous trouvez ci après les règlements chèques saisies '
        "au niveau du\r\nmodule avance médecin qui ne figurent pas dans le module "
        'financier&nbsp;: </p><p class="MsoNormal">B 05/ DECAISSEMENT</p>'
        "Centre Médical de réeducation : 11 248.000 dt , chèque n° 4066678.<br>"
        "Centre Médical de réeducation : 8 260.250 dt , chèque n° 4066679.<br><br>"
        "Ainsi que le problème de retard dans l'intégration des règlements : des "
        "règlements validés dès le 18/08/2014 ne figurent pas dans le module financier."
        "<br><br>Cordialement<br>Contact Test<br>\r\n"
    ),
    "module_designation": "Financier",
    "objectif_reclamation": "Assistance",
}

# Cas de référence : gros bloc MSO/Word en tête + plusieurs occurrences de
# "( à supprimer )" (contenu client légitime, à ne PAS toucher).
TICKET_5638 = {
    "idMessage": 1003,
    "objet": "problèmes du module financier  , à  collaborateur test",
    "message": (
        "<!--[if gte mso 9]><xml>\r\n <o:OfficeDocumentSettings>\r\n  <o:RelyOnVML/>\r\n"
        "  <o:AllowPNG/>\r\n </o:OfficeDocumentSettings>\r\n</xml><![endif]-->\r\n\r\n"
        '<p class="MsoNormal">Bonjour</p>'
        '<p class="MsoNormal">Veuillez trouver ci après les problèmes du module '
        "financier non encore résolus :<br></p>"
        '<p class="MsoNormal">Décaissement B05&nbsp;:</p>'
        '<p class="MsoNormal">Préavis CHQ n° 5181407 du 10/11/2014 d’un montant de '
        "292.200 dt n’est pas rapproché et n’existe pas à la banque.</p>"
        '<p class="MsoNormal">Chèque n°4066626/300 dt rapproché , existe encore '
        "dans la trésorerie</p>"
        '<p class="MsoNormal">Ecriture fictive pour régulariser un dossier d’un '
        "montant de 17&nbsp;980.797 dt et 0.060 millimes ( à supprimer )</p>"
        '<p class="MsoNormal">Des virements déjà réalisés à la banque : virement '
        "499&nbsp;646.000 dt /100 000.000 dt /89 752.994 dt ( à supprimer )</p>"
        '<p class="MsoNormal">bordereau de versement chèque 14007729/47 500dt '
        "réalisé à la banque avant le démarrage du module financier (à supprimer )</p>"
        '<p class="MsoNormal">Cordialement</p><p class="MsoNormal">Contact Test<br></p>'
        "<!--[if gte mso 9]><xml>\r\n <w:WordDocument>\r\n  <w:View>Normal</w:View>\r\n"
        " </w:WordDocument>\r\n</xml><![endif]-->"
    ),
    "module_designation": "Financier",
    "objectif_reclamation": "Assistance",
}

# Fixture synthétique : nombreux <br> et \r\n internes au message d'origine
# (non-régression Bug A.2 : texte_complet/longueur_texte ne doivent jamais
# rester vides à cause de retours à la ligne non normalisés).
TICKET_MANY_LINEBREAKS = {
    "idMessage": 1004,
    "objet": "Problème module financier",
    "message": (
        "Bonjour<br>Ligne 1 du message<br><br>Ligne 2 du message\r\n\r\n"
        "Ligne 3 du message<br>\r\nLigne 4<br><br><br>Ligne 5 finale<br>"
    ),
    "module_designation": "Financier",
    "objectif_reclamation": "Assistance",
}

# Fixture synthétique : suffixe collaborateur avec PLUSIEURS destinataires
# séparés par ";"
TICKET_MULTI_RECIPIENT = {
    "idMessage": 1005,
    "objet": "Identifiant de remise , à  collaborateur alpha;collaborateur beta",
    "message": "<p>Merci de créer un identifiant de remise pour ce dossier client.</p>",
    "module_designation": "Financier",
    "objectif_reclamation": "Assistance",
}


def _df(*rows):
    return pd.DataFrame(list(rows))


def test_normalize_category_strips_case_and_spaces():
    assert normalize_category("Facturation  ") == "facturation"
    assert normalize_category("  Comptabilité") == "comptabilité"
    assert normalize_category(None) == ""


def test_outlook_ticket_is_cleaned_and_kept():
    df = _df(TICKET_OUTLOOK_FACTURATION)
    kept, report = process(df, min_length=25)

    assert len(kept) == 1
    row = kept.iloc[0]
    assert "OfficeDocumentSettings" not in row["message_clean"]
    assert "MsoNormal" not in row["message_clean"]
    assert "2024-458" in row["message_clean"]
    # Signature retirée par anonymize_text
    assert "Amal Trabelsi" not in row["message_clean"]
    assert "Cordialement" not in row["message_clean"]
    # Correctif à la source : la valeur stockée est strippée (casse/accents
    # conservés), plus d'espaces superflus hérités de l'export SQL.
    assert row["module_designation"] == "Facturation"
    assert row["longueur_texte"] == len(row["texte_complet"])
    assert report[report["idMessage"] == 1001].iloc[0]["statut_filtre"] == "retenu"


def test_base64_image_is_neutralized_and_ticket_kept():
    df = _df(TICKET_WITH_BASE64_IMAGE)
    kept, report = process(df, min_length=25)

    assert len(kept) == 1
    row = kept.iloc[0]
    assert "[IMAGE]" in row["message_clean"]
    assert "base64" not in row["message_clean"]
    assert "iVBORw0KGgo" not in row["message_clean"]


def test_out_of_scope_module_is_rejected():
    df = _df(TICKET_OUT_OF_SCOPE_MODULE)
    kept, report = process(df, min_length=25)

    assert len(kept) == 0
    reason = report[report["idMessage"] == 1003].iloc[0]["raison_rejet"]
    assert "module hors périmètre" in reason


def test_stored_category_values_are_stripped_at_source():
    # Correctif à la source : module_designation / objectif_reclamation stockés
    # dans le CSV doivent être strippés (l'export SQL laisse des espaces en fin
    # type "Facturation  "/"Comptabilité " qui cassaient le bucketing aval par
    # égalité). Casse et accents conservés.
    ticket = {
        "idMessage": 9100,
        "objet": "changement de nom du patient sur la facture",
        "message": "Bonjour, merci de corriger le nom du patient relatif au dossier.",
        "module_designation": "Facturation  ",       # 2 espaces en fin
        "objectif_reclamation": "  Assistance ",       # espaces avant/après
    }
    kept, _ = process(_df(ticket), min_length=25)
    assert len(kept) == 1
    row = kept.iloc[0]
    assert row["module_designation"] == "Facturation"
    assert row["objectif_reclamation"] == "Assistance"


def test_too_short_text_is_rejected():
    df = _df(TICKET_TOO_SHORT)
    kept, report = process(df, min_length=25)

    assert len(kept) == 0
    reason = report[report["idMessage"] == 1004].iloc[0]["raison_rejet"]
    assert "trop court" in reason


def test_exact_duplicates_are_flagged_not_removed():
    dup = dict(TICKET_OUTLOOK_FACTURATION)
    dup["idMessage"] = 1005
    df = _df(TICKET_OUTLOOK_FACTURATION, dup)
    kept, report = process(df, min_length=25)

    # Les deux tickets identiques sont conservés tous les deux (pas de suppression),
    # mais marqués avec le même hash_doublon pour un dédoublonnage sémantique en aval.
    assert len(kept) == 2
    assert kept.iloc[0]["texte_complet"] == kept.iloc[1]["texte_complet"]
    assert kept.iloc[0]["hash_doublon"] == kept.iloc[1]["hash_doublon"]


def test_output_columns_match_spec():
    df = _df(TICKET_OUTLOOK_FACTURATION)
    kept, report = process(df, min_length=25)

    assert list(kept.columns) == [
        "idMessage", "module_designation", "objectif_reclamation", "IdClient",
        "objet_clean", "message_clean", "texte_complet", "longueur_texte", "hash_doublon",
    ]
    assert list(report.columns) == ["idMessage", "statut_filtre", "raison_rejet"]


def test_idclient_present_even_when_missing_from_source():
    # La requête SQL corrigée n'exporte plus IdClient : la colonne doit quand
    # même exister en sortie (valeur vide), jamais être supprimée silencieusement.
    df = _df(TICKET_TOO_SHORT)  # ne fournit pas IdClient
    kept, report = process(df, min_length=0)

    assert "IdClient" in kept.columns
    assert len(kept) == 1
    assert pd.isna(kept.iloc[0]["IdClient"])


# ── Non-régression Bug 1 (CSV) + Bug 2 (nettoyage métier) ──────────────────────

def test_3926_collaborator_name_leak_is_removed_and_row_has_9_columns(tmp_path):
    df = _df(TICKET_3926)
    kept, report = process(df, min_length=25)

    row = kept.iloc[0]
    assert "collaborateur" not in row["objet_clean"]
    assert "test" not in row["objet_clean"]
    assert row["objet_clean"] == "problème de rapprochement au niveau du module financier"

    # Round-trip CSV réel (pas juste l'objet DataFrame en mémoire) : 9 colonnes,
    # IdClient et hash_doublon présents, aucun décalage dû à la virgule dans objet_clean.
    out = tmp_path / "reclamations_seules.csv"
    kept.to_csv(out, index=False, encoding="utf-8-sig")
    reread = pd.read_csv(out, encoding="utf-8-sig")

    expected_cols = [
        "idMessage", "module_designation", "objectif_reclamation", "IdClient",
        "objet_clean", "message_clean", "texte_complet", "longueur_texte", "hash_doublon",
    ]
    assert list(reread.columns) == expected_cols
    assert len(reread.columns) == 9
    assert reread.iloc[0]["idMessage"] == 1001
    assert "IdClient" in reread.columns
    assert "hash_doublon" in reread.columns
    assert pd.notna(reread.iloc[0]["hash_doublon"])


def test_multi_recipient_collaborator_suffix_is_fully_removed():
    # Plusieurs destinataires séparés par ";" dans le suffixe système
    df = _df(TICKET_MULTI_RECIPIENT)
    kept, report = process(df, min_length=0)

    row = kept.iloc[0]
    assert row["objet_clean"] == "Identifiant de remise"
    assert "collaborateur" not in row["objet_clean"]
    assert "alpha" not in row["objet_clean"]
    assert "beta" not in row["objet_clean"]


def test_many_internal_linebreaks_do_not_empty_texte_complet(tmp_path):
    # Non-régression Bug A.2 : de nombreux <br>/\r\n internes au message ne
    # doivent jamais laisser texte_complet ou longueur_texte vides.
    df = _df(TICKET_MANY_LINEBREAKS)
    kept, report = process(df, min_length=25)

    assert len(kept) == 1
    row = kept.iloc[0]
    assert row["texte_complet"] != ""
    assert pd.notna(row["texte_complet"])
    assert row["longueur_texte"] == len(row["texte_complet"])
    assert "\n" not in row["texte_complet"]
    assert "\r" not in row["texte_complet"]
    for fragment in ("Ligne 1", "Ligne 2", "Ligne 3", "Ligne 4", "Ligne 5 finale"):
        assert fragment in row["texte_complet"]

    # Round-trip CSV réel : la cellule ne doit pas casser l'écriture/lecture
    out = tmp_path / "reclamations_seules.csv"
    kept.to_csv(out, index=False, encoding="utf-8-sig")
    reread = pd.read_csv(out, encoding="utf-8-sig")
    assert len(reread.columns) == 9
    assert pd.notna(reread.iloc[0]["texte_complet"])
    assert reread.iloc[0]["longueur_texte"] == len(reread.iloc[0]["texte_complet"])


def test_4116_business_numbers_are_genericized():
    df = _df(TICKET_4116)
    kept, report = process(df, min_length=25)

    row = kept.iloc[0]
    assert "[REF]" in row["texte_complet"]
    assert "[MONTANT]" in row["texte_complet"]
    assert "4066678" not in row["texte_complet"]
    assert "11 248.000" not in row["texte_complet"]
    # Suffixe collaborateur retiré de l'objet malgré le préfixe "*"
    assert "collaborateur" not in row["objet_clean"]


def test_5638_a_supprimer_is_never_touched():
    df = _df(TICKET_5638)
    kept, report = process(df, min_length=25)

    row = kept.iloc[0]
    # 3 occurrences dans la fixture : "( à supprimer )" x2 et "(à supprimer )" x1
    assert row["message_clean"].count("supprimer") == 3
    assert "MsoNormal" not in row["message_clean"]
    assert "OfficeDocumentSettings" not in row["message_clean"]
    assert "WordDocument" not in row["message_clean"]
    # Références/montants du même message bien génériciés au passage
    assert "[REF]" in row["message_clean"]
    assert "[MONTANT]" in row["message_clean"]


def test_full_regenerated_csv_has_exactly_9_columns_every_row(tmp_path):
    """Régénère le CSV à partir des 3 tickets réels et vérifie qu'aucune ligne
    ne déborde sur les colonnes suivantes (Bug 1 : virgules dans objet_clean)."""
    df = _df(TICKET_3926, TICKET_4116, TICKET_5638)
    kept, report = process(df, min_length=25)

    out = tmp_path / "reclamations_seules.csv"
    kept.to_csv(out, index=False, encoding="utf-8-sig")
    reread = pd.read_csv(out, encoding="utf-8-sig")

    assert len(reread.columns) == 9
    assert len(reread) == len(kept) == 3
    assert reread["texte_complet"].notna().all()
    assert (reread["longueur_texte"] == reread["texte_complet"].str.len()).all()
    assert list(reread["idMessage"]) == [1001, 1002, 1003]
