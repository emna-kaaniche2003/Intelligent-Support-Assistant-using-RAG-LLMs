import pytest
import numpy as np
from unittest.mock import MagicMock

from ingest_qr_technique import (
    build_texte_embedding,
    build_payload,
    build_point_id,
    prepare_entries,
    validate_entry,
    run,
)


@pytest.fixture
def config():
    return {
        "chemin_source": "fake_source.json",
        "collection_name": "connaissances_modules",
        "module": "facturation",
        "source_type": "qr_technique",
        "modele_embedding": "paraphrase-multilingual-mpnet-base-v2",
        "longueur_troncature_reponse": 400,
        "taille_batch": 2,
    }


def make_entry(**overrides):
    entry = {
        "id": 1,
        "categorie": "Listes",
        "source": "Interface",
        "ecran": "listeDossiersInstances",
        "question": "Comment chercher un dossier ?",
        "reponse": "Cliquez sur Rechercher.",
        "mots_cles": ["recherche", "dossier"],
    }
    entry.update(overrides)
    return entry


class TestTexteEmbedding:
    def test_formule_exacte_sans_troncature(self):
        entry = make_entry()
        result = build_texte_embedding(entry, troncature=400)
        expected = (
            "Listes. Comment chercher un dossier ?. Comment chercher un dossier ?. "
            "Cliquez sur Rechercher."
        )
        assert result == expected

    def test_formule_exacte_avec_troncature(self):
        long_reponse = "A" * 500
        entry = make_entry(reponse=long_reponse)
        result = build_texte_embedding(entry, troncature=400)
        expected = f"Listes. Comment chercher un dossier ?. Comment chercher un dossier ?. {'A' * 400}"
        assert result == expected

    def test_pas_de_troncature_si_reponse_plus_courte(self):
        short_reponse = "Réponse courte"
        entry = make_entry(reponse=short_reponse)
        result = build_texte_embedding(entry, troncature=400)
        assert result.endswith(short_reponse)
        assert len(short_reponse) < 400


class TestPayload:
    def test_payload_contient_interface_ou_serveur_pas_source(self, config):
        entry = make_entry(source="Serveur")
        payload = build_payload(entry, config)
        assert "interface_ou_serveur" in payload
        assert payload["interface_ou_serveur"] == "Serveur"
        assert "source" not in payload

    def test_payload_contient_module_et_source_type_depuis_config(self, config):
        entry = make_entry()
        payload = build_payload(entry, config)
        assert payload["module"] == "facturation"
        assert payload["source_type"] == "qr_technique"

    def test_payload_reponse_texte_complet_non_tronque(self, config):
        long_reponse = "B" * 500
        entry = make_entry(reponse=long_reponse)
        payload = build_payload(entry, config)
        assert payload["reponse"] == long_reponse
        assert len(payload["reponse"]) == 500

    def test_payload_mots_cles_conserves(self, config):
        entry = make_entry(mots_cles=["a", "b", "c"])
        payload = build_payload(entry, config)
        assert payload["mots_cles"] == ["a", "b", "c"]


class TestPointId:
    def test_id_deterministe_meme_entree_meme_id(self):
        id1 = build_point_id(42, "qr_technique")
        id2 = build_point_id(42, "qr_technique")
        assert id1 == id2

    def test_id_est_uuid_valide(self):
        import uuid
        point_id = build_point_id(42, "qr_technique")
        # Ne lève pas d'exception si c'est un UUID valide
        uuid.UUID(point_id)

    def test_id_differe_selon_source_type(self):
        """Namespace par source_type évite collision entre sources différentes."""
        id_qr = build_point_id(1, "qr_technique")
        id_doc = build_point_id(1, "doc_module")
        assert id_qr != id_doc

    def test_id_differe_selon_id_origine(self):
        id1 = build_point_id(1, "qr_technique")
        id2 = build_point_id(2, "qr_technique")
        assert id1 != id2


class TestValidation:
    def test_entree_valide_ne_leve_pas(self):
        entry = make_entry()
        validate_entry(entry)  # ne doit pas lever

    def test_champ_manquant_leve_exception(self):
        entry = make_entry()
        del entry["mots_cles"]
        with pytest.raises(ValueError, match="mots_cles"):
            validate_entry(entry)


class TestPrepareEntriesErrorIsolation:
    """Une entrée malformée est journalisée et n'empêche pas le traitement du reste."""

    def test_entree_malformee_isolee_reste_traite(self, config):
        entry_ok_1 = make_entry(id=1)
        entry_malformee = make_entry(id=2)
        del entry_malformee["question"]
        entry_ok_2 = make_entry(id=3)

        raw_entries = [entry_ok_1, entry_malformee, entry_ok_2]
        valid_entries, texts, errors = prepare_entries(raw_entries, config)

        assert len(valid_entries) == 2
        assert len(texts) == 2
        assert len(errors) == 1
        assert errors[0][0] == 2  # id de l'entrée fautive journalisé

    def test_toutes_valides_aucune_erreur(self, config):
        raw_entries = [make_entry(id=1), make_entry(id=2)]
        valid_entries, texts, errors = prepare_entries(raw_entries, config)
        assert len(valid_entries) == 2
        assert len(errors) == 0

    def test_toutes_malformees_liste_vide_sans_crash(self, config):
        entry1 = make_entry(id=1)
        del entry1["ecran"]
        entry2 = make_entry(id=2)
        del entry2["categorie"]

        valid_entries, texts, errors = prepare_entries([entry1, entry2], config)
        assert len(valid_entries) == 0
        assert len(errors) == 2


class TestBatching:
    """Vérifie que le batching respecte taille_batch (appels encodeur + upsert)."""

    def test_nombre_appels_encode_correspond_au_nombre_de_batches(self, config, monkeypatch):
        # 5 entrées valides, taille_batch=2 → 3 batches (2, 2, 1)
        raw_entries = [make_entry(id=i) for i in range(5)]

        mock_load_config = lambda path: config
        mock_load_source = lambda path: raw_entries

        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]] * 10  # taille suffisante pour tout batch

        mock_client = MagicMock()

        monkeypatch.setattr("ingest_qr_technique.load_config", mock_load_config)
        monkeypatch.setattr("ingest_qr_technique.load_source", mock_load_source)
        monkeypatch.setattr("ingest_qr_technique.SentenceTransformer", lambda name: mock_model)
        monkeypatch.setattr("ingest_qr_technique.QdrantClient", lambda url, timeout: mock_client)

        class FakeSettings:
            qdrant_url = "http://localhost:6333"

        monkeypatch.setattr("ingest_qr_technique.get_settings", lambda: FakeSettings())

        # encode_dense_batch appelle model.encode une fois par batch → wrap pour renvoyer
        # la bonne taille de liste selon la taille du batch reçu
        def fake_encode(batch_texts, **kwargs):
            return np.array([[0.1, 0.2, 0.3] for _ in batch_texts])

        mock_model.encode.side_effect = fake_encode

        run(config_path="fake_config.json")

        # 3 batches attendus (tailles 2, 2, 1) → 3 appels encode, 3 appels upsert
        assert mock_model.encode.call_count == 3
        assert mock_client.upsert.call_count == 3

    def test_upsert_appele_avec_bonne_collection(self, config, monkeypatch):
        raw_entries = [make_entry(id=1)]

        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda batch_texts, **kwargs: np.array([[0.1, 0.2] for _ in batch_texts])
        mock_client = MagicMock()

        monkeypatch.setattr("ingest_qr_technique.load_config", lambda path: config)
        monkeypatch.setattr("ingest_qr_technique.load_source", lambda path: raw_entries)
        monkeypatch.setattr("ingest_qr_technique.SentenceTransformer", lambda name: mock_model)
        monkeypatch.setattr("ingest_qr_technique.QdrantClient", lambda url, timeout: mock_client)

        class FakeSettings:
            qdrant_url = "http://localhost:6333"

        monkeypatch.setattr("ingest_qr_technique.get_settings", lambda: FakeSettings())

        run(config_path="fake_config.json")

        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "connaissances_modules"

    def test_entree_malformee_n_empeche_pas_upsert_du_reste(self, config, monkeypatch, capsys):
        entry_ok = make_entry(id=1)
        entry_malformee = make_entry(id=2)
        del entry_malformee["mots_cles"]

        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda batch_texts, **kwargs: np.array([[0.1, 0.2] for _ in batch_texts])
        mock_client = MagicMock()

        monkeypatch.setattr("ingest_qr_technique.load_config", lambda path: config)
        monkeypatch.setattr("ingest_qr_technique.load_source", lambda path: [entry_ok, entry_malformee])
        monkeypatch.setattr("ingest_qr_technique.SentenceTransformer", lambda name: mock_model)
        monkeypatch.setattr("ingest_qr_technique.QdrantClient", lambda url, timeout: mock_client)

        class FakeSettings:
            qdrant_url = "http://localhost:6333"

        monkeypatch.setattr("ingest_qr_technique.get_settings", lambda: FakeSettings())

        run(config_path="fake_config.json")

        # Le point valide doit quand même être upserté
        mock_client.upsert.assert_called_once()
        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == 1

        # L'erreur doit être journalisée (stderr)
        captured = capsys.readouterr()
        assert "id=2" in captured.err
