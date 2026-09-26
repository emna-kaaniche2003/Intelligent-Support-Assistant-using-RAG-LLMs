"""
tests/test_extract_and_filter.py
Non-régression du refactoring de extract_and_filter.py : après extraction de la
logique de nettoyage vers cleaning_utils.py, les mêmes seuils et critères de
rétention doivent produire les mêmes décisions de filtrage qu'avant.
"""

import pandas as pd

from data_pipeline.extract_and_filter import apply_cleaning_and_rules, apply_col_map

MIN_DESC = 80
MIN_RES = 100

VALID_DESC = "x" * MIN_DESC
VALID_RES = "y" * MIN_RES


def _row(titre="Titre", description=VALID_DESC, resolution=VALID_RES):
    return {"titre": titre, "description": description, "resolution": resolution}


def test_valid_ticket_is_kept():
    df = pd.DataFrame([_row()])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 1
    assert len(rejected) == 0


def test_empty_field_after_html_cleanup_is_rejected():
    df = pd.DataFrame([_row(description="<p></p>")])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 0
    assert "vide après nettoyage HTML" in rejected.iloc[0]["raison_rejet"]


def test_short_description_is_rejected():
    df = pd.DataFrame([_row(description="trop court")])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 0
    assert f"< {MIN_DESC} chars" in rejected.iloc[0]["raison_rejet"]


def test_short_resolution_is_rejected():
    df = pd.DataFrame([_row(resolution="trop court")])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 0
    assert f"< {MIN_RES} chars" in rejected.iloc[0]["raison_rejet"]


def test_duplicate_description_is_rejected():
    df = pd.DataFrame([_row(), _row(titre="Autre titre")])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 1
    assert len(rejected) == 1
    assert "doublon" in rejected.iloc[0]["raison_rejet"]


def test_security_content_is_rejected():
    df = pd.DataFrame([_row(resolution=VALID_RES + " mot de passe: hunter2xyz")])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 0
    assert "sensible" in rejected.iloc[0]["raison_rejet"]


def test_vague_resolution_is_rejected():
    # min_res=0 pour isoler la règle "résolution vague" du filtre de longueur
    # (les patterns vagues sont par nature courts, donc déjà couverts par
    # min_res=100 en usage réel — cf. test_short_resolution_is_rejected).
    df = pd.DataFrame([_row(resolution="Résolu")])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, min_res=0)
    assert len(kept) == 0
    assert "vague" in rejected.iloc[0]["raison_rejet"]


def test_html_is_stripped_before_length_check():
    # Un texte de 80+ caractères de balises HTML mais peu de texte réel doit
    # être rejeté pour longueur insuffisante après nettoyage, pas conservé.
    padded_html = "<div>" + "<b></b>" * 20 + "court</div>"
    df = pd.DataFrame([_row(description=padded_html)])
    kept, rejected = apply_cleaning_and_rules(df, MIN_DESC, MIN_RES)
    assert len(kept) == 0
    assert f"< {MIN_DESC} chars" in rejected.iloc[0]["raison_rejet"]


def test_optional_categorical_columns_are_stripped_at_source():
    # Correctif à la source : les colonnes catégorielles optionnelles stockées
    # (module/priorite/type_priorite/statut) doivent être strippées des espaces
    # superflus hérités de l'export SQL (même classe de bug que le pipeline FAQ).
    df = pd.DataFrame([{
        "titre": "Titre",
        "description": VALID_DESC,
        "resolution": VALID_RES,
        "module": "Comptabilité ",         # espace en fin
        "priorite": "  Anomalie/Bug  ",     # espaces avant/après + interne
        "statut": "Résolu ",
    }])
    out = apply_col_map(df, col_map=None)
    assert out.iloc[0]["module"] == "Comptabilité"
    assert out.iloc[0]["priorite"] == "Anomalie/Bug"
    assert out.iloc[0]["statut"] == "Résolu"
