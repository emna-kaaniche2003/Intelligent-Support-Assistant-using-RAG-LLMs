import pytest
from unittest.mock import MagicMock, call
from qdrant_client.models import Distance, PayloadSchemaType

from create_collection import (
    create_collection,
    create_payload_indexes,
    collection_already_exists,
    resolve_distance,
    resolve_payload_schema,
    run,
)


@pytest.fixture
def config():
    return {
        "host": "localhost",
        "port": 6333,
        "collection_name": "connaissances_modules",
        "dense_vector_name": "dense",
        "dense_vector_size": 768,
        "distance": "Cosine",
        "sparse_vector_name": "bm25",
        "payload_indexes": [
            {"field": "module", "type": "keyword"},
            {"field": "source_type", "type": "keyword"},
            {"field": "categorie", "type": "keyword"},
            {"field": "type_contenu", "type": "keyword"},
        ],
    }


@pytest.fixture
def mock_client():
    return MagicMock()


class TestCreateCollection:
    """Validation de la création de collection avec vecteurs dense + sparse."""

    def test_collection_creee_avec_bon_nom(self, mock_client, config):
        create_collection(mock_client, config)
        _, kwargs = mock_client.create_collection.call_args
        assert kwargs["collection_name"] == "connaissances_modules"

    def test_vecteur_dense_bonne_taille_et_distance(self, mock_client, config):
        create_collection(mock_client, config)
        _, kwargs = mock_client.create_collection.call_args
        dense_params = kwargs["vectors_config"]["dense"]
        assert dense_params.size == 768
        assert dense_params.distance == Distance.COSINE

    def test_vecteur_sparse_bon_nom(self, mock_client, config):
        create_collection(mock_client, config)
        _, kwargs = mock_client.create_collection.call_args
        sparse_config = kwargs["sparse_vectors_config"]
        assert "bm25" in sparse_config

    def test_create_collection_appelee_une_seule_fois(self, mock_client, config):
        create_collection(mock_client, config)
        mock_client.create_collection.assert_called_once()


class TestPayloadIndexes:
    """Validation des index de payload."""

    def test_quatre_index_crees(self, mock_client, config):
        created = create_payload_indexes(mock_client, config)
        assert mock_client.create_payload_index.call_count == 4
        assert created == ["module", "source_type", "categorie", "type_contenu"]

    def test_index_avec_bon_nom_de_champ(self, mock_client, config):
        create_payload_indexes(mock_client, config)
        called_fields = [
            kwargs["field_name"]
            for _, kwargs in mock_client.create_payload_index.call_args_list
        ]
        assert called_fields == ["module", "source_type", "categorie", "type_contenu"]

    def test_index_type_keyword(self, mock_client, config):
        create_payload_indexes(mock_client, config)
        for _, kwargs in mock_client.create_payload_index.call_args_list:
            assert kwargs["field_schema"] == PayloadSchemaType.KEYWORD

    def test_index_sur_bonne_collection(self, mock_client, config):
        create_payload_indexes(mock_client, config)
        for _, kwargs in mock_client.create_payload_index.call_args_list:
            assert kwargs["collection_name"] == "connaissances_modules"


class TestCollectionExistanteSansForceRecreate:
    """Si la collection existe et --force-recreate n'est pas passé, on ne touche à rien."""

    def test_arret_sans_appel_delete_ni_create(self, mock_client, config):
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=42)

        result = collection_already_exists(mock_client, "connaissances_modules", force_recreate=False)

        assert result is True
        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()

    def test_run_ne_cree_rien_si_existe_et_pas_force(self, mock_client, config, monkeypatch):
        monkeypatch.setattr("create_collection.load_config", lambda path: config)
        monkeypatch.setattr("create_collection.connect_client", lambda host, port: mock_client)
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=10)

        run(config_path="fake_path.json", force_recreate=False)

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()
        mock_client.create_payload_index.assert_not_called()


class TestCollectionExistanteAvecForceRecreate:
    """Si --force-recreate est passé et la collection existe, delete puis create dans cet ordre."""

    def test_delete_puis_create_dans_cet_ordre(self, mock_client, config):
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=5)

        result = collection_already_exists(mock_client, "connaissances_modules", force_recreate=True)

        assert result is False
        mock_client.delete_collection.assert_called_once_with("connaissances_modules")

    def test_run_supprime_et_recree_avec_force(self, mock_client, config, monkeypatch):
        monkeypatch.setattr("create_collection.load_config", lambda path: config)
        monkeypatch.setattr("create_collection.connect_client", lambda host, port: mock_client)
        mock_client.collection_exists.return_value = True
        mock_client.count.return_value = MagicMock(count=5)

        run(config_path="fake_path.json", force_recreate=True)

        mock_client.delete_collection.assert_called_once_with("connaissances_modules")
        mock_client.create_collection.assert_called_once()
        assert mock_client.create_payload_index.call_count == 4

        # Vérifier l'ordre : delete_collection appelé avant create_collection
        delete_call_index = None
        create_call_index = None
        for i, mock_call in enumerate(mock_client.method_calls):
            if mock_call[0] == "delete_collection":
                delete_call_index = i
            if mock_call[0] == "create_collection":
                create_call_index = i
        assert delete_call_index is not None
        assert create_call_index is not None
        assert delete_call_index < create_call_index


class TestCollectionInexistante:
    """Si la collection n'existe pas, elle est créée normalement (avec ou sans --force-recreate)."""

    def test_collection_inexistante_retourne_false(self, mock_client, config):
        mock_client.collection_exists.return_value = False

        result = collection_already_exists(mock_client, "connaissances_modules", force_recreate=False)

        assert result is False
        mock_client.delete_collection.assert_not_called()

    def test_run_cree_collection_inexistante(self, mock_client, config, monkeypatch):
        monkeypatch.setattr("create_collection.load_config", lambda path: config)
        monkeypatch.setattr("create_collection.connect_client", lambda host, port: mock_client)
        mock_client.collection_exists.return_value = False

        run(config_path="fake_path.json", force_recreate=False)

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_called_once()
        assert mock_client.create_payload_index.call_count == 4


class TestResolveHelpers:
    """Validation des fonctions de résolution distance/type."""

    def test_resolve_distance_cosine(self):
        assert resolve_distance("Cosine") == Distance.COSINE

    def test_resolve_distance_invalide_leve_exception(self):
        with pytest.raises(ValueError, match="inconnue"):
            resolve_distance("InvalidDistance")

    def test_resolve_payload_schema_keyword(self):
        assert resolve_payload_schema("keyword") == PayloadSchemaType.KEYWORD

    def test_resolve_payload_schema_invalide_leve_exception(self):
        with pytest.raises(ValueError, match="non supporté"):
            resolve_payload_schema("invalid_type")
