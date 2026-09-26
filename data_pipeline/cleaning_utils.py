"""
data_pipeline/cleaning_utils.py
Fonctions de nettoyage et normalisation partagées par les pipelines data :
  - extract_and_filter.py  (réclamations + résolutions -> tickets_resolus)
  - extract_faq_source.py  (réclamations seules -> reclamations_seules)
"""

from __future__ import annotations

import hashlib
import html
import re

from bs4 import BeautifulSoup, Comment

# Commentaires conditionnels MSO/Word : <!--[if gte mso 9]>...<![endif]-->
_MSO_CONDITIONAL_RE = re.compile(r"<!--\s*\[if[^\]]*\]>.*?<!\[endif\]-->", re.IGNORECASE | re.DOTALL)

# data URI d'image encodée en base64 : data:image/png;base64,AAAA...
_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", re.IGNORECASE)

# Blocs base64 orphelins (hors data URI) : longue séquence alphabet base64 contiguë
# candidate -> validée ensuite par _looks_like_base64_blob (diversité de caractères)
# pour éviter les faux positifs sur du texte naturel (ex. longs identifiants).
_ORPHAN_BASE64_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")


def _looks_like_base64_blob(candidate: str) -> bool:
    if len(candidate) < 80:
        return False
    has_upper = any(c.isupper() for c in candidate)
    has_lower = any(c.islower() for c in candidate)
    has_digit = any(c.isdigit() for c in candidate)
    return has_upper and has_lower and has_digit and len(set(candidate)) >= 20


def _replace_orphan_base64(match: re.Match) -> str:
    return " [IMAGE] " if _looks_like_base64_blob(match.group()) else match.group()

_WHITESPACE_RE = re.compile(r"\s+")

# Formules de politesse/signature : tout ce qui suit est considéré comme signature
# "merci d'avance" ajouté après vérification sur un cas représentatif
# 6113/18436/25474/34256) : toujours suivi du nom du signataire en fin de
# message, jamais en milieu de phrase dans ce corpus (à ne pas confondre avec
# "merci de <verbe>", une formule de demande très fréquente et jamais une
# signature — non ajoutée à cette liste).
_SIGNATURE_RE = re.compile(
    r"(?i)\b(cordialement|bien\s+à\s+vous|bien\s+cordialement|salutations?\s+distingu[ée]es?|"
    r"meilleures?\s+salutations?|sincèrement|respectueusement|merci\s+d['’]avance)\b.*$",
    re.DOTALL,
)

# Téléphones tunisiens : séquences de 8 chiffres consécutifs
_PHONE_RE = re.compile(r"(?<!\d)\d{8}(?!\d)")

# Suffixe système en fin de champ "objet" : ", à <destinataire[;destinataire...]>"
# ou "... : à <destinataire...>" (variante avec ":", observée sur quelques
# tickets). Un ou plusieurs employés destinataires, séparés par ";". Ancré en
# fin de chaîne pour ne jamais toucher du contenu client légitime comme
# "( à supprimer )" (pas précédé d'une virgule/deux-points, pas en fin).
_NAME_WORD = r"[A-Za-zÀ-ÖØ-öø-ÿ]+"
_NAME_GROUP = rf"{_NAME_WORD}(?:[\s-]{_NAME_WORD}){{0,2}}"
_COLLABORATOR_SUFFIX_RE = re.compile(
    rf"[,:]\s*à\s+{_NAME_GROUP}(?:;\s*{_NAME_GROUP}){{0,4}}\s*$",
    re.IGNORECASE,
)

# Label spaCy pour les entités "personne" en français : "PER" (pas "PERSON"
# comme dans les modèles anglais).
_PERSON_ENTITY_LABEL = "PER"
_PERSON_MASK_TAG = "[PERSONNE]"

# Singleton paresseux : les modèles spaCy ne sont chargés (import + spacy.load)
# qu'au premier appel réel nécessitant le masquage des personnes, jamais à
# l'import de ce module. Cache par nom de modèle (utile pour les tests qui
# surchargent ner_model).
_NLP_MODEL_CACHE: dict[str, object] = {}


def _load_person_ner_model(model_name: str):
    """
    Charge (et met en cache) le pipeline spaCy `model_name`.
    Fail-closed : si le modèle est absent/non installé, lève une exception
    explicite plutôt que de laisser passer un texte non masqué en silence.
    """
    if model_name not in _NLP_MODEL_CACHE:
        import spacy

        try:
            _NLP_MODEL_CACHE[model_name] = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"Masquage des personnes activé (ENABLE_PERSON_ANONYMIZATION) mais le "
                f"modèle spaCy '{model_name}' est introuvable/non installé. "
                "Voir requirements.txt (wheel fr_core_news_sm) et check_env.py. "
                "Refus explicite de traiter le texte sans masquage (fail-closed)."
            ) from exc
    return _NLP_MODEL_CACHE[model_name]


# Vocabulaire métier (chèque/bordereau/décaissement/virement...) que
# fr_core_news_sm étiquette parfois à tort comme PER dans les tickets où
# clean_html() a aplati plusieurs paragraphes en un seul bloc de texte sans
# ponctuation de séparation (ex. "Préavis CHQ n° 5181407",
# "Chèque n°4066626" détectés comme PER par le modèle). Ces mots ne sont
# jamais des noms propres dans ce corpus — les mêmes mots-types que
# _BUSINESS_REF_RE, qui seront de toute façon généricisés en [REF] par
# genericize_business_numbers() en aval si on ne les masque pas ici en amont.
# Complété (balayage complet du corpus, cf. skill.md section 10) par deux
# sous-catégories DÉLIBÉRÉMENT closes/énumérables (contrairement au jargon
# métier ci-dessus, qui reste un problème ouvert accepté — voir docstring) :
#   - formules de politesse/requête formelle ("veuillez", "prière") : jeu
#     fermé et bien identifié en français, ne grossira pas.
#   - noms des 12 mois de l'année.
#   - "clinisys" : nom de la plateforme elle-même (cf. README.md), revient
#     fréquemment dans le corpus, jamais un nom de personne.
_PERSON_MASK_DENYLIST = {
    # vocabulaire métier (chèque/bordereau/...)
    "cheque", "cheques", "chq", "bordereau", "bord",
    "preavis", "decaissement", "virement", "ecriture",
    # plateforme
    "clinisys",
    # formules de politesse/requête formelle (jeu fermé, pas du jargon métier)
    "veuillez", "priere",
    # mois de l'année (jeu fermé)
    "janvier", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "aout", "septembre", "octobre", "novembre", "decembre",
}
_ACCENT_MAP = str.maketrans("èéêàâ", "eeeaa")

# Tags déjà posés par une étape antérieure (signature/collaborateur/téléphone,
# ou par un appel précédent en cas de ré-exécution) : [TEL], [REF], [MONTANT],
# [PERSONNE], [IMAGE]... Découvert sur un cas représentatif,
# "chèque [TEL]/104 dt") : quand un tel tag est collé sans espace à de la
# ponctuation/des chiffres, spaCy peut produire une entité PER qui chevauche
# PARTIELLEMENT le tag (ex. "TEL]/104"), corrompant le texte en "[[PERSONNE] dt"
# au lieu de "[TEL]/104 dt". Toute entité qui chevauche un tag existant est donc
# ignorée — protection nécessaire à l'idempotence et à la non-interférence.
_EXISTING_TAG_RE = re.compile(r"\[[A-ZÀ-Ö]+\]")

# Ponctuation "dure" jamais présente à l'intérieur d'un nom réel dans ce
# corpus (contrairement à l'apostrophe ou au trait d'union, légitimes dans
# des noms comme "N'Diaye"/"Jean-Paul"). Découvert sur données réelles : sur
# du texte sans espace après un signe de ponctuation (courant dans les
# tickets tapés vite, ou après un tag déjà posé), spaCy produit parfois une
# entité PER qui déborde sur ce qui suit — ex. "NOMTEST;il",
# "JOINTES.MERCI", "TEL]/104" (même famille
# que le bug [[PERSONNE] déjà corrigé par _overlaps_existing_tag, mais qui ne
# couvre que NOS PROPRES tags — cette règle est plus générale et couvre aussi
# la ponctuation déjà présente dans le texte source). Contrepartie acceptée :
# une entité légitime mais accolée sans espace à une abréviation de titre
# (ex. "Mr.Amine") n'est alors pas non plus masquée — limite documentée,
# préférée à un texte corrompu.
_HARD_PUNCTUATION_RE = re.compile(r"[.;\[\]()/]")

# Longueur minimale (en lettres) pour qu'une entité soit un nom plausible.
# Découvert sur données réelles : spaCy étiquette parfois PER un fragment
# d'une seule lettre issu d'une élision/abréviation collée sans espace —
# "n'" (dans "n'est pas", "n'existe pas"), "N" (dans "N°1888"), "N+1" (notation
# comptable "exercice suivant"). Aucun nom réel observé dans ce corpus ne fait
# moins de 3 lettres ; seuil choisi en conséquence, pas une valeur arbitraire.
_MIN_PERSON_NAME_LETTERS = 3


def _looks_like_business_term(span_text: str) -> bool:
    for word in re.findall(r"[^\W\d_]+", span_text, re.UNICODE):
        if word.casefold().translate(_ACCENT_MAP) in _PERSON_MASK_DENYLIST:
            return True
    return False


def _overlaps_existing_tag(start: int, end: int, tag_spans: list[tuple[int, int]]) -> bool:
    return any(start < tag_end and end > tag_start for tag_start, tag_end in tag_spans)


def _is_too_short_to_be_a_name(span_text: str) -> bool:
    return sum(1 for c in span_text if c.isalpha()) < _MIN_PERSON_NAME_LETTERS


def _mask_person_entities(text: str, model_name: str) -> str:
    if not text:
        return text
    nlp = _load_person_ner_model(model_name)
    doc = nlp(text)
    tag_spans = [m.span() for m in _EXISTING_TAG_RE.finditer(text)]
    spans = [
        (ent.start_char, ent.end_char)
        for ent in doc.ents
        if ent.label_ == _PERSON_ENTITY_LABEL
        and not _looks_like_business_term(ent.text)
        and not _overlaps_existing_tag(ent.start_char, ent.end_char, tag_spans)
        and not _is_too_short_to_be_a_name(ent.text)
        and not _HARD_PUNCTUATION_RE.search(ent.text)
    ]
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + _PERSON_MASK_TAG + text[end:]
    return text


# ── Masquage structurel des noms MAJUSCULES en contexte de modification ───────
# fr_core_news_sm ne détecte PAS les noms écrits entièrement en majuscules
# (motif réel et fréquent des tickets "changement de nom du patient", sous-thème
# modifications_donnees du module Facturation — le plus gros du corpus, ~29
# tickets concernés, noms patients santé fuitant à 100 % avant ce correctif).
# Cette règle structurelle complète le NER.
#
# Elle N'est déclenchée QUE si le texte contient un déclencheur contextuel de
# modification de données. Ce gâte est PORTEUR, pas décoratif : mesuré sur le
# corpus complet, la même détection de casse SANS déclencheur produit 71 % de
# faux positifs (jargon comptable majuscule type "DECAISSEMENT REG EFFET SMEDI",
# "BANQUE EERONE", formules "BONNE RECEPTION"...) ; AVEC le déclencheur, le seul
# faux positif résiduel est un nom de société (cf. limites du docstring).
#
# Déclencheurs en constante module (cohérent avec _SIGNATURE_RE/_BUSINESS_REF_RE,
# jamais dans config.py — seuls le flag et le modèle NER y vivent).
_MODIF_TRIGGER_RE = re.compile(
    r"(?i)(?:"
    r"nom du patient|nom de la patiente|changer le nom|modifier le nom|"
    r"changement de nom|changer l['’]ancien|au lieu de|nom de la st[ée]|"
    r"nom de la soci[ée]t[ée]|l['’]intervenant|date de naissance|"
    r"modifier le|changer le"
    r")"
)

# Un "token majuscule" : soit un mot d'au moins 2 lettres majuscules (accents
# inclus, apostrophe/trait d'union internes tolérés — "N'DIAYE", "BEN-ALI"),
# soit une initiale isolée (1 lettre majuscule — "ZANE D I" où D et I sont des
# fragments d'un nom coupé par des balises HTML). Un "run" est une suite de tels
# tokens séparés par des espaces (les noms composés tunisiens type
# "BEN HAMID DALILA EP BAABAA" sont ainsi capturés d'un bloc).
_UPPER = r"A-ZÀ-ÖØ-Þ"
_CAPS_MULTI = rf"[{_UPPER}]{{2,}}(?:['’\-][{_UPPER}]+)*"
_CAPS_TOKEN = rf"(?:{_CAPS_MULTI}|[{_UPPER}])"
_CAPS_NAME_RUN_RE = re.compile(rf"\b{_CAPS_TOKEN}(?:\s+{_CAPS_TOKEN})*\b")

# Un run doit totaliser au moins 4 lettres majuscules ET contenir au moins un
# mot de >= 2 lettres, pour ne pas masquer des acronymes courts ou des suites
# d'initiales seules ("A B C") ; seuil calé sur le corpus (aucun vrai nom
# masquable en dessous).
_MIN_UPPERCASE_NAME_LETTERS = 4


def _mask_uppercase_names_in_context(text: str) -> str:
    """
    Masque les runs de casse majuscule (noms patients/sociétés) UNIQUEMENT dans
    les tickets contenant un déclencheur contextuel de modification de données.
    Complète _mask_person_entities pour le cas des noms tout-majuscules que
    fr_core_news_sm ne détecte pas. Ne touche jamais les tags déjà posés.
    """
    if not text or not _MODIF_TRIGGER_RE.search(text):
        return text

    tag_spans = [m.span() for m in _EXISTING_TAG_RE.finditer(text)]
    spans = []
    for m in _CAPS_NAME_RUN_RE.finditer(text):
        run = m.group()
        if sum(1 for c in run if c.isalpha()) < _MIN_UPPERCASE_NAME_LETTERS:
            continue
        if not any(len(word) >= 2 for word in run.split()):
            continue
        if _overlaps_existing_tag(m.start(), m.end(), tag_spans):
            continue
        spans.append(m.span())

    for start, end in sorted(spans, reverse=True):
        text = text[:start] + _PERSON_MASK_TAG + text[end:]
    return text


# Références chèque/bordereau/CHQ : mot désignant le type de document
# (chèque, bord, bordereau, CHQ) suivi de "n°"/"N°" puis de chiffres.
# Le mot-type est OBLIGATOIRE pour ne pas génériciser d'autres numéros
# (ex. numéro de facture) qui ne sont pas dans le périmètre demandé.
_BUSINESS_REF_RE = re.compile(
    r"(?i)\b(?:ch[eè]que|bordereau|bord|chq)\b\.?\s*n\s*°\s*\d+"
)

# Montants : chiffres avec séparateur décimal (espaces tolérés comme séparateur
# de milliers), devise dt/DT/TND optionnelle.
_MONTANT_RE = re.compile(r"(?i)\b\d[\d\s]*[.,]\s*\d+\s*(?:dt|tnd)?\b")


def clean_html(text: str) -> str:
    """
    Nettoie un texte HTML brut :
      - neutralise les images base64 (data URI ou blocs orphelins) -> [IMAGE]
      - retire les commentaires conditionnels MSO/Word
      - retire les balises <style>/<xml> et leur contenu
      - retire les balises HTML restantes (BeautifulSoup)
      - décode les entités HTML (&nbsp;, &eacute;, &#233;...)
      - normalise espaces/retours à la ligne
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    raw = text
    raw = _DATA_URI_RE.sub(" [IMAGE] ", raw)
    raw = _ORPHAN_BASE64_CANDIDATE_RE.sub(_replace_orphan_base64, raw)
    raw = _MSO_CONDITIONAL_RE.sub(" ", raw)

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup.find_all("img"):
        tag.replace_with(" [IMAGE] ")
    for tag in soup.find_all(["style", "xml"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    text = soup.get_text(separator=" ")

    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def anonymize_text(
    text: str,
    mask_persons: bool | None = None,
    ner_model: str | None = None,
) -> str:
    """
    Retire les signatures usuelles (et tout ce qui suit), le suffixe système
    ", à <collaborateur>" en fin de champ, masque les numéros de téléphone
    tunisiens (8 chiffres consécutifs) par [TEL], puis masque les noms de
    personnes détectés (entités PER via spaCy) par [PERSONNE].
    Suppose un texte déjà nettoyé (post clean_html).

    mask_persons / ner_model : surcharges explicites (tests, injection de
    dépendance). Laissés à None en usage normal — la valeur est alors lue
    depuis ai_engine.config.get_settings() (ENABLE_PERSON_ANONYMIZATION,
    PERSON_NER_MODEL), jamais hardcodée.

    Limites connues du masquage des personnes (réduction de risque, PAS une
    anonymisation certifiée à 100%) :
      - Rappel imparfait sur les noms tunisiens peu représentés dans le corpus
        d'entraînement du modèle français standard (fr_core_news_sm, entraîné
        sur du français de référence).
      - fr_core_news_sm ne fournit pas de score de confiance exploitable par
        entité : aucun seuil de confiance réglable n'est donc appliqué ici.
      - Noms écrits ENTIÈREMENT EN MAJUSCULES (ex. citations lors d'un
        changement de nom patient) : souvent NON
        détectés comme PER par ce modèle (étiquetés ORG/MISC, ou ratés) —
        vérifié sur données réelles. Une réparation heuristique (reparsing en
        casse mixte) a été testée puis écartée : elle introduit de faux
        positifs sur du vocabulaire métier (ex. "MODULE FINANCIER" ->
        "Financier" étiqueté à tort PER) et une couverture incomplète sur les
        noms composés ("EPOUSE ZAYANI" non rattaché). Limite acceptée en
        l'état, documentée plutôt que masquée par une fausse garantie.
      - Faux positifs sur du texte sans ponctuation de séparation entre
        phrases (aplatissement HTML par clean_html()) : neutralisés par
        trois garde-fous, tous vérifiés sur un balayage complet du corpus
        (85 entités PER brutes sur reclamations_seules.csv, cf. skill.md
        section 10) :
          1. _PERSON_MASK_DENYLIST : vocabulaire métier (chèque/bordereau/
             CHQ/préavis/décaissement/virement/écriture), formules de
             politesse/requête formelle ("veuillez", "prière" — jeu FERMÉ,
             contrairement au jargon métier ci-dessous) et mois de l'année.
          2. _is_too_short_to_be_a_name (< 3 lettres) : élimine les fragments
             d'une lettre issus d'une élision/abréviation collée sans espace
             ("n'" dans "n'est pas"/"n'existe pas", "N" dans "N°1888", "N+1").
             Règle structurelle, pas un mot-à-mot — corrige le même défaut
             que l'"n'" sans qu'il faille l'énumérer.
          3. _HARD_PUNCTUATION_RE (`.;[]()/`) : rejette toute entité dont le
             texte contient une ponctuation qui n'apparaît jamais dans un nom
             réel de ce corpus (contrairement à l'apostrophe/au trait d'union,
             légitimes) — corrige "NAOUREZ;il", "JOINTES.MERCI", "TEL]/104".
             Contrepartie acceptée : un nom réel collé sans espace à un titre
             abrégé (ex. "Mr.Amine") n'est alors pas non plus masqué.
        Résidu explicitement NON traité par denylist, par choix assumé (voir
        recommandation ci-dessous) : jargon métier/comptable ouvert (ex.
        "Timbre", "HT", "Paie", "Compte Libellé", codes "REG*"/"folio*") et
        noms de sociétés étiquetés PER au lieu d'ORG (ex. "Echifa Holding").
      - Pourquoi ce jargon métier n'est PAS ajouté au denylist : contrairement
        au vocabulaire métier ci-dessus (chèque/bordereau/...) qui est un jeu
        fermé et déjà central au domaine, les termes comptables/codes internes
        sont un problème ouvert (un nouveau terme apparaît à chaque nouveau
        ticket) — un denylist mot-à-mot deviendrait vite ingérable et ne
        convergerait jamais. Le filet de sécurité prévu pour ce résidu est la
        revue humaine par l'équipe support déjà prévue dans le pipeline FAQ
        avant rédaction des réponses types et ingestion RAG (cf. CLAUDE.md),
        pas un correctif automatique supplémentaire.

    Masquage structurel des noms tout-MAJUSCULES (_mask_uppercase_names_in_context,
    appelé après le NER) — complète fr_core_news_sm qui ne détecte pas ces noms
    (motif "changement de nom du patient", sous-thème modifications_donnees) :
      - Déclencheurs contextuels volontairement un peu génériques ("au lieu de",
        "modifier le", "changer le") : leur généricité est CONTENUE empiriquement
        par l'exigence de casse majuscule + l'exclusion des tags — mesuré sur le
        corpus complet, aucun faux positif réel au-delà des sociétés (ci-dessous).
        Sans le déclencheur, la seule détection de casse produirait 71 % de faux
        positifs (jargon comptable majuscule) : le gâte est donc porteur.
      - Sociétés mal étiquetées : un nom de société tout-majuscules dans un
        ticket de modification (ex. "SOCOMENIN INDUSTRIAL CONSTRUCTION",
        un cas de référence) est masqué [PERSONNE] au lieu d'un éventuel
        [ORGANISATION] — label imprécis, mais l'info identifiante est quand même
        retirée. Pas de tag [ORGANISATION] séparé dans cette itération (décision
        explicite) ; couvert par la revue humaine.
      - Noms tout-MAJUSCULES HORS contexte de modification de données (ex. un
        message entier tapé en capitales, "L'ATTENTION DE MR YACINE FAKHFEKH…")
        ne sont PAS couverts : pas de déclencheur -> non masqués. Pattern
        distinct, hors périmètre de ce correctif.
      - Une revue humaine reste nécessaire avant tout usage externe du texte.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    if mask_persons is None or ner_model is None:
        from ai_engine.config import get_settings

        settings = get_settings()
        if mask_persons is None:
            mask_persons = settings.enable_person_anonymization
        if ner_model is None:
            ner_model = settings.person_ner_model

    text = _SIGNATURE_RE.sub("", text)
    text = _COLLABORATOR_SUFFIX_RE.sub("", text)
    text = _PHONE_RE.sub("[TEL]", text)
    if mask_persons:
        text = _mask_person_entities(text, ner_model)
        # Complément structurel : noms tout-MAJUSCULES que le NER rate, en
        # contexte de modification de données uniquement (pure regex, ne
        # nécessite pas le modèle spaCy). Appliqué APRÈS le NER pour ne pas
        # ré-attraper les [PERSONNE] déjà posés (garde _overlaps_existing_tag).
        text = _mask_uppercase_names_in_context(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def genericize_business_numbers(text: str) -> str:
    """
    Généricise les identifiants métier (numéros de chèque/bordereau, montants)
    pour faciliter la détection de quasi-doublons sémantiques en aval : deux
    tickets identiques sauf le numéro de chèque doivent matcher.
    Séparée d'anonymize_text : objectif différent (clustering, pas vie privée).
    Utilisée uniquement par extract_faq_source.py, pas extract_and_filter.py.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    text = _BUSINESS_REF_RE.sub("[REF]", text)
    text = _MONTANT_RE.sub("[MONTANT]", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_noise(text: str, min_length: int) -> bool:
    """
    Retourne True si le texte est vide ou trop court (< min_length) après
    nettoyage, donc non exploitable pour le pipeline.
    """
    if not isinstance(text, str):
        return True
    return len(text.strip()) < min_length


def hash_text(text: str) -> str:
    """Hash MD5 normalisé (minuscules, espaces retirés) pour détection de doublons exacts."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    normalized = re.sub(r"\s+", "", text.lower())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()
