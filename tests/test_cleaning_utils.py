"""
tests/test_cleaning_utils.py
Tests unitaires pour data_pipeline/cleaning_utils.py — fonctions de nettoyage
partagées par extract_and_filter.py et extract_faq_source.py.
"""

from data_pipeline.cleaning_utils import (
    anonymize_text,
    clean_html,
    genericize_business_numbers,
    hash_text,
    is_noise,
    _mask_uppercase_names_in_context,
)


# ── clean_html ────────────────────────────────────────────────────────────────

def test_clean_html_strips_basic_tags():
    raw = "<p>Bonjour, <b>voici</b> le problème.</p>"
    assert clean_html(raw) == "Bonjour, voici le problème."


def test_clean_html_removes_img_with_marker():
    raw = "Voir la pièce jointe <img src='photo.png'> ci-dessous."
    result = clean_html(raw)
    assert "[IMAGE]" in result
    assert "<img" not in result


def test_clean_html_decodes_entities():
    raw = "Probl&egrave;me r&eacute;solu&nbsp;: cong&eacute;s"
    result = clean_html(raw)
    assert "Problème résolu : congés" == result


def test_clean_html_strips_mso_conditional_comments():
    raw = (
        "<!--[if gte mso 9]><xml><o:OfficeDocumentSettings>"
        "</o:OfficeDocumentSettings></xml><![endif]-->"
        "<p>Texte utile du ticket.</p>"
    )
    result = clean_html(raw)
    assert "OfficeDocumentSettings" not in result
    assert "Texte utile du ticket." == result


def test_clean_html_strips_style_and_xml_tags():
    raw = "<style>.MsoNormal { margin: 0; }</style><p>Contenu réel</p>"
    result = clean_html(raw)
    assert "MsoNormal" not in result
    assert result == "Contenu réel"


def test_clean_html_neutralizes_base64_data_uri():
    raw = (
        "<p>Capture :</p>"
        "<img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA\">"
    )
    result = clean_html(raw)
    assert "[IMAGE]" in result
    assert "base64" not in result
    assert "iVBORw0KGgo" not in result


def test_clean_html_neutralizes_orphan_base64_block():
    orphan = (
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdA"
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdA=="
    )
    raw = f"Voici le fichier corrompu : {orphan} fin du message"
    result = clean_html(raw)
    assert "[IMAGE]" in result
    assert orphan not in result


def test_clean_html_normalizes_whitespace():
    raw = "<p>Ligne 1</p>\n\n<p>   Ligne   2   </p>"
    assert clean_html(raw) == "Ligne 1 Ligne 2"


def test_clean_html_handles_non_string_input():
    assert clean_html(None) == ""
    assert clean_html(123) == "123"


# ── anonymize_text ────────────────────────────────────────────────────────────

def test_anonymize_text_removes_signature():
    text = "Le problème persiste sur le module facturation. Cordialement, Ahmed Ben Ali"
    result = anonymize_text(text)
    assert "Cordialement" not in result
    assert "Ahmed" not in result
    assert result.startswith("Le problème persiste")


def test_anonymize_text_masks_phone_number():
    text = "Merci de me rappeler au 12345678 dès que possible."
    result = anonymize_text(text)
    assert "12345678" not in result
    assert "[TEL]" in result


def test_anonymize_text_ignores_short_numeric_sequences():
    text = "Le ticket numéro 1234 est bloqué."
    result = anonymize_text(text)
    assert "1234" in result
    assert "[TEL]" not in result


def test_anonymize_text_removes_collaborator_suffix():
    # Suffixe système ", à <collaborateur>"
    text = "problème de rapprochement au niveau du module financier , à  collaborateur test"
    result = anonymize_text(text)
    assert "collaborateur" not in result
    assert "test" not in result
    assert result == "problème de rapprochement au niveau du module financier"


def test_anonymize_text_collaborator_suffix_handles_multiple_words():
    assert anonymize_text("Question sur le module , à collaborateur alpha") == "Question sur le module"
    assert anonymize_text("Bug facturation , à collaborateur beta") == "Bug facturation"


def test_anonymize_text_collaborator_suffix_handles_multiple_recipients():
    text = "le financier , à  collaborateur alpha;collaborateur beta;collaborateur gamma"
    result = anonymize_text(text)
    assert result == "le financier"
    for name in ("alpha", "beta", "gamma"):
        assert name not in result

    text2 = "Identifiant de remise , à  collaborateur delta;collaborateur epsilon"
    assert anonymize_text(text2) == "Identifiant de remise"


def test_anonymize_text_collaborator_suffix_handles_colon_variant():
    assert anonymize_text("Réclamation Facturation. : à  collaborateur zeta") == "Réclamation Facturation."
    text = "INTÉGRATION VENTE /  FACTURE N°12345678 : à  collaborateur test"
    result = anonymize_text(text)
    assert "collaborateur" not in result
    assert result.startswith("INTÉGRATION VENTE")


def test_anonymize_text_does_not_touch_mise_a_jour():
    # Non-régression : "Mise à jour" n'est ni précédé d'une virgule ni de deux-points
    assert anonymize_text("Mise à jour du plan comptable") == "Mise à jour du plan comptable"


def test_anonymize_text_does_not_touch_a_supprimer():
    # Non-régression : "( à supprimer )" est du contenu client légitime,
    # jamais précédé d'une virgule et jamais en fin de champ -> ne pas toucher.
    text = "Écriture fictive à régulariser ( à supprimer ) merci de vérifier"
    result = anonymize_text(text)
    assert "( à supprimer )" in result


def test_anonymize_text_does_not_touch_a_supprimer_mid_sentence_variants():
    text = "réalisé à la banque avant le démarrage du module financier (à supprimer )"
    result = anonymize_text(text)
    assert "supprimer" in result


def test_anonymize_text_signature_removes_merci_davance_variant():
    text = "Merci de vérifier ce point pour le contrôleur de cnss. Merci d'avance PersonneTest"
    result = anonymize_text(text)
    assert "Merci d'avance" not in result
    assert "PersonneTest" not in result
    assert result.startswith("Merci de vérifier ce point pour le contrôleur de cnss.")


# ── anonymize_text : masquage des personnes (PER, spaCy NER) ──────────────────
# ENABLE_PERSON_ANONYMIZATION est actif par défaut (filet de sécurité) : les
# tests ci-dessus (signature/suffixe/téléphone) tournent donc aussi avec le
# masquage NER actif, ce qui a été vérifié ne rien casser (aucun de ces
# extraits ne contient de nom détecté par le modèle).

def test_anonymize_text_masks_person_name_fixture():
    text = "Jean Exemple : [MONTANT] , [REF]."
    result = anonymize_text(text)
    assert result == "[PERSONNE] : [MONTANT] , [REF]."


def test_anonymize_text_masks_all_occurrences_fixture():
    text = (
        "Jean Exemple : [MONTANT] , [REF]. Jean Exemple : [MONTANT] , [REF]. "
        "Jean Exemple : [MONTANT] , [REF]."
    )
    result = anonymize_text(text)
    assert "Jean" not in result
    assert "Exemple" not in result
    assert result.count("[PERSONNE]") == 3


def test_anonymize_text_masks_person_name_in_raw_sentence():
    text = (
        "Vous trouvez ci après les règlements chèques saisies au niveau du "
        "module avance médecin : Jean Exemple : 12 000.000 dt , chèque n° 1234567."
    )
    result = anonymize_text(text)
    assert "Jean" not in result
    assert "Exemple" not in result
    assert "[PERSONNE]" in result


def test_anonymize_text_does_not_mask_organization_name_fixture():
    # "Centre Médical de réeducation" est un nom
    # d'organisme (ORG), pas une personne (PER) — non masqué par design (voir
    # décision documentée : masquage PER uniquement dans cette itération).
    text = "Centre Médical de réeducation : [MONTANT] , [REF]."
    result = anonymize_text(text)
    assert "Centre Médical de réeducation" in result
    assert "[PERSONNE]" not in result


def test_anonymize_text_known_limitation_partial_recall_on_repeated_name():
    text = (
    "B 05/ DECAISSEMENT Jean Exemple : [MONTANT] , [REF]. Jean Exemple : [MONTANT] , [REF]. "
    "B02/DECAISSEMENT Jean Exemple :[MONTANT] , [REF]."
    )
    result = anonymize_text(text)
    assert result.count("[PERSONNE]") == 1
    assert "B02/DECAISSEMENT Jean Exemple" in result


def test_anonymize_text_no_person_name_is_not_altered():
    # Non-régression : un texte sans nom de personne ne doit pas être modifié
    # par le nouveau traitement.
    text = "Le module financier affiche une erreur de rapprochement bancaire."
    assert anonymize_text(text) == text


def test_anonymize_text_does_not_reinterpret_existing_tags_as_names():
    # Non-interférence : [REF]/[MONTANT]/[TEL] déjà posés ne doivent pas être
    # altérés ni re-détectés comme des noms de personnes.
    text = "Merci de rappeler au [TEL] concernant [MONTANT] et [REF]."
    result = anonymize_text(text)
    assert result == text


def test_anonymize_text_person_masking_is_idempotent():
    text = "Jean Exemple : 12 000.000 dt , chèque n° 1234567."
    once = anonymize_text(text)
    twice = anonymize_text(once)
    assert twice == once


# ── anonymize_text : masquage des noms MAJUSCULES en contexte de modification ──
# fr_core_news_sm ne détecte PAS les noms écrits tout en majuscules (motif réel
# des tickets "changement de nom du patient", sous-thème modifications_donnees
# du module Facturation — le plus gros du corpus). Une règle structurelle
# dédiée (_mask_uppercase_names_in_context) complète le NER : dans un ticket
# contenant un déclencheur contextuel, elle masque les runs de casse majuscule.

def test_anonymize_text_masks_allcaps_patient_name_fixture():
    text = 'Merci de changer le nom du patient par "PATIENT TEST EPOUSE EXEMPLE" au lieu de "PATIENT TEST".'
    result = anonymize_text(text)
    assert "PATIENT" not in result
    assert "TEST" not in result
    assert "EXEMPLE" not in result
    assert "[PERSONNE]" in result


def test_anonymize_text_masks_allcaps_single_word_name_fixture():
    text = (
        "Merci de nous Modifier le de la patiente relative à l'admission N° [TEL] : "
        '"NOMTEST" au lieu de "NOMTESTB".'
    )
    result = anonymize_text(text)
    assert "NOMTEST" not in result
    assert "NOMTESTB" not in result
    assert result.count("[PERSONNE]") == 2


def test_anonymize_text_masks_allcaps_name_with_isolated_initials_fixture():
    # Nom coupé par des balises HTML de mise en forme -> après clean_html
    # "NOMTEST" devient "NOM T E" (initiales isolées).
    text = (
        'Merci de changer le nom du patient du dossier [TEL] par "NOM T E" '
        'au lieu de "NOMTEST" relative à la facture [TEL].'
    )
    result = anonymize_text(text)
    assert "NOM" not in result
    assert "NOMTEST" not in result
    assert "[PERSONNE]" in result


def test_anonymize_text_masks_allcaps_multiword_name_fixture():
    # Deux noms composés en majuscules.
    text = (
        "Bonjour, Merci de changer le nom du patient du dossier [TEL] par "
        '" PATIENT TEST EPOUSE EXEMPLE " au lieu de " EXEMPLE TEST EPOUSE PATIENT " '
        "relative à la facture [TEL]. Salutations."
    )
    result = anonymize_text(text)
    for token in ("PATIENT", "TEST", "EXEMPLE", "EPOUSE"):
        assert token not in result
    assert result.count("[PERSONNE]") == 2


def test_anonymize_text_uppercase_rule_requires_context_trigger():
    # Anti-faux-positif : un run
    # de casse majuscule de jargon comptable ("DECAISSEMENT REG EFFET SMEDI",
    # "BANQUE EERONE") NE DOIT PAS être masqué en l'absence de déclencheur
    # contextuel de modification de données. Le gâte-déclencheur est porteur :
    # sans lui, cette règle serait un gouffre à faux positifs (mesuré : 71 %).
    text = (
        "B04/DECAISSEMENT REG EFFET SMEDI à vérifier quel module. "
        "ce virement est intégré dans le compte b01 (BANQUE EERONE ) la banque correcte b05"
    )
    result = anonymize_text(text)
    assert "[PERSONNE]" not in result
    assert "REG EFFET SMEDI" in result
    assert "BANQUE EERONE" in result


def test_anonymize_text_uppercase_rule_ignores_normal_case_near_trigger():
    # Déclencheur présent MAIS contenu à proximité en casse normale (pas de run
    # majuscule) -> la règle structurelle ne doit rien masquer. Cas synthétique
    # (aucun run majuscule dans le corpus réel avec déclencheur qui ne soit un
    # nom). On teste la fonction structurelle isolément pour ne pas dépendre du
    # comportement spaCy sur un vrai nom en casse normale.
    text = "Merci de modifier le paramètre du module au lieu de la valeur actuelle."
    assert _mask_uppercase_names_in_context(text) == text


def test_anonymize_text_uppercase_rule_does_not_touch_existing_tags():
    # Non-interférence : les tags déjà posés ([TEL]/[REF]/[MONTANT]/[PERSONNE])
    # ne doivent jamais être ré-attrapés comme un run majuscule, même quand un
    # déclencheur est présent dans le texte.
    text = "Merci de changer le nom du patient du dossier [TEL] : [PERSONNE] , [MONTANT] , [REF]."
    result = anonymize_text(text)
    assert "[TEL]" in result
    assert "[MONTANT]" in result
    assert "[REF]" in result
    assert "[[PERSONNE]" not in result
    assert "[PERSONNE]PERSONNE" not in result


def test_anonymize_text_uppercase_rule_is_idempotent():
    text = (
        "Merci de changer le nom du patient du dossier [TEL] par "
        '" PATIENT TEST EPOUSE EXEMPLE " au lieu de " EXEMPLE TEST EPOUSE PATIENT ".'
    )
    once = anonymize_text(text)
    twice = anonymize_text(once)
    assert twice == once


def test_anonymize_text_uppercase_rule_disabled_via_flag():
    # Flag désactivé -> aucun run majuscule masqué (comportement d'avant).
    text = 'Merci de changer le nom du patient par "NOMTEST" au lieu de "NOMTESTB".'
    result = anonymize_text(text, mask_persons=False)
    assert "NOMTEST" in result
    assert "NOMTESTB" in result
    assert "[PERSONNE]" not in result


def test_anonymize_text_does_not_mask_business_vocabulary_as_person():
    # Sur du texte MSO aplati sans ponctuation de
        # Sur du texte MSO aplati sans ponctuation de
    # séparation, fr_core_news_sm étiquette parfois à tort le vocabulaire
    # métier (chèque/CHQ/préavis) comme PER. Ce vocabulaire ne doit jamais
    # être masqué, et doit rester disponible pour genericize_business_numbers()
    # en aval.
    text = "Préavis CHQ n° 5181407 du 10/11/2014 : Chèque n°4066626/300 dt rapproché"
    result = anonymize_text(text)
    assert "[PERSONNE]" not in result
    assert "Préavis" in result
    assert "Chèque" in result


def test_anonymize_text_does_not_corrupt_tag_glued_to_digits():
    # "chèque 12345678/104 dt" -> le numéro de
        # "chèque 12345678/104 dt" -> le numéro de
    # 8 chiffres est masqué [TEL] par la règle téléphone AVANT le masquage des
    # personnes. Sur ce texte, spaCy détectait une entité PER chevauchant
    # partiellement le tag ("TEL]/104"), corrompant le résultat en
    # "[[PERSONNE] dt" au lieu de "[TEL]/104 dt". Non-régression sur ce bug.
    text = "bordereau de versement chèque 14002470/104 dt réalisé à la banque"
    result = anonymize_text(text)
    assert "[[PERSONNE]" not in result
    assert "[TEL]/104 dt" in result


def test_anonymize_text_does_not_corrupt_negation_apostrophe():
    # "n'" (élision de "ne") est un fragment d'une
    # lettre, jamais un nom de personne. Étaient corrompues en
    # "[PERSONNE]est"/"[PERSONNE]existe" avant le garde-fou de longueur
    # minimale (_is_too_short_to_be_a_name).
    assert (
        anonymize_text("d'un montant de 292.200 dt n'est pas rapproché")
        == "d'un montant de 292.200 dt n'est pas rapproché"
    )
    assert (
        anonymize_text("n'existe pas à la banque.")
        == "n'existe pas à la banque."
    )
    assert (
        anonymize_text("n'existe pas dans la trésorerie")
        == "n'existe pas dans la trésorerie"
    )


def test_anonymize_text_does_not_mask_politeness_request_verbs():
    # "Veuillez"/"Prière" sont
    # des formules de politesse/requête standard, jamais des noms de personne.
    assert anonymize_text("Veuillez trouver ci après les problèmes") == (
        "Veuillez trouver ci après les problèmes"
    )
    assert anonymize_text("Prière faire le paramétrage de ce compte") == (
        "Prière faire le paramétrage de ce compte"
    )
    result = anonymize_text("Bonjour Veuillez trouver ci joint l'extrait bancaire")
    assert "[PERSONNE]" not in result


def test_anonymize_text_rejects_entities_with_hard_internal_punctuation():
    # Une entité PER dont le texte
    # contient une ponctuation "dure" (jamais présente dans un vrai nom de ce
    # corpus) est rejetée plutôt que de produire un texte corrompu.
    text = "BONJOUR NOMTEST;il y a un probleme ds L'integration"
    assert anonymize_text(text) == text

    text2 = "BJR TEST, VOILA LES PIECES JOINTES.MERCI."
    assert anonymize_text(text2) == text2


def test_anonymize_text_person_masking_disabled_via_flag():
    text = "Jean Exemple : 12 000.000 dt , chèque n° 1234567."
    result = anonymize_text(text, mask_persons=False)
    assert result == text
    assert "[PERSONNE]" not in result


def test_anonymize_text_person_masking_fails_closed_when_model_unavailable():
    # Fail-closed : si le modèle spaCy demandé est absent/non chargeable et que
    # le masquage est actif, la fonction doit lever une exception explicite —
    # jamais renvoyer silencieusement un texte non masqué.
    import pytest

    text = "Jean Exemple : 12 000.000 dt , chèque n° 1234567."
    with pytest.raises(Exception):
        anonymize_text(text, mask_persons=True, ner_model="__modele_spacy_inexistant__")


# ── genericize_business_numbers ───────────────────────────────────────────────

def test_genericize_replaces_cheque_reference():
    assert genericize_business_numbers("chèque n° 4066678") == "[REF]"
    assert genericize_business_numbers("CHQ N° 8878818") == "[REF]"
    assert genericize_business_numbers("bord n° 140005647") == "[REF]"
    assert genericize_business_numbers("bordereau n°140007235") == "[REF]"


def test_genericize_replaces_montant():
    assert genericize_business_numbers("16 007.093 TND") == "[MONTANT]"
    assert genericize_business_numbers("292.200 dt") == "[MONTANT]"
    assert genericize_business_numbers("11 248.000 dt") == "[MONTANT]"


def test_genericize_real_ticket_4116_excerpt():
    text = "Centre Médical de réeducation : 11 248.000 dt , chèque n° 4066678."
    result = genericize_business_numbers(text)
    assert "[MONTANT]" in result
    assert "[REF]" in result
    assert "4066678" not in result
    assert "248.000" not in result


def test_genericize_is_not_applied_by_extract_and_filter_pipeline():
    # extract_and_filter.py (tickets_resolus) ne doit pas importer cette fonction
    import ast
    from pathlib import Path

    source = Path("data_pipeline/extract_and_filter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "genericize_business_numbers" not in imported_names


# ── is_noise ──────────────────────────────────────────────────────────────────

def test_is_noise_true_for_empty_text():
    assert is_noise("", 25) is True
    assert is_noise("   ", 25) is True


def test_is_noise_true_for_short_text():
    assert is_noise("trop court", 25) is True


def test_is_noise_false_for_sufficient_text():
    assert is_noise("x" * 25, 25) is False


def test_is_noise_threshold_is_parametrized():
    text = "x" * 30
    assert is_noise(text, 25) is False
    assert is_noise(text, 80) is True


# ── hash_text ─────────────────────────────────────────────────────────────────

def test_hash_text_is_deterministic():
    assert hash_text("Bonjour le monde") == hash_text("Bonjour le monde")


def test_hash_text_ignores_case_and_whitespace():
    assert hash_text("Bonjour   le Monde") == hash_text("bonjourlemonde")


def test_hash_text_differs_for_different_content():
    assert hash_text("texte A") != hash_text("texte B")
