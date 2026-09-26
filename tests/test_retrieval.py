import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import ScoredPoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.rag.evaluate_retrieval import (
    evaluate_question,
    evaluate_question_rerank,
    hybrid_search,
    reciprocal_rank_fusion,
    resolve_golden_dataset,
    run,
)
from data_pipeline.ingestion.ingest_qr_technique import build_point_id


def make_scored_point(point_id: str, score: float = 1.0) -> ScoredPoint:
    return ScoredPoint(id=point_id, version=0, score=score, payload={}, vector=None)


class TestReciprocalRankFusion:
    """Fusion RRF de deux classements dense/sparse."""

    def test_point_uniquement_en_dense(self):
        dense = [make_scored_point("a"), make_scored_point("b")]
        sparse = []
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        ids = [pid for pid, _ in fused]
        assert ids[0] == "a"
        assert ids[1] == "b"

    def test_point_present_dans_les_deux_listes_score_plus_haut(self):
        dense = [make_scored_point("a"), make_scored_point("b")]
        sparse = [make_scored_point("b"), make_scored_point("c")]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        scores = dict(fused)
        # "b" apparaît dans les deux listes → score cumulé plus élevé que "a" ou "c" seuls
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_tri_par_score_decroissant(self):
        dense = [make_scored_point("x"), make_scored_point("y"), make_scored_point("z")]
        sparse = [make_scored_point("z"), make_scored_point("y")]
        fused = reciprocal_rank_fusion(dense, sparse, k=60)
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)

    def test_poids_dense_nul_annule_la_contribution_dense(self):
        """w_dense=0.0 : les points présents uniquement en dense reçoivent un
        score nul, seul le classement sparse contribue au score fusionné."""
        dense = [make_scored_point("a"), make_scored_point("b")]
        sparse = [make_scored_point("c"), make_scored_point("d")]
        fused = reciprocal_rank_fusion(dense, sparse, k=60, w_dense=0.0, w_sparse=1.0)
        scores = dict(fused)
        assert scores["a"] == 0.0
        assert scores["b"] == 0.0
        assert scores["c"] > 0.0
        assert scores["d"] > 0.0

    def test_poids_asymetrique_favorise_le_classement_le_plus_pondere(self):
        """Un poids dense plus élevé augmente l'importance relative d'un point
        présent uniquement en dense par rapport à un point présent uniquement
        en sparse (même rang dans leur liste respective)."""
        dense = [make_scored_point("y"), make_scored_point("x")]
        sparse = [make_scored_point("z"), make_scored_point("x")]

        fused_egal = dict(reciprocal_rank_fusion(dense, sparse, k=60, w_dense=0.5, w_sparse=0.5))
        fused_dense_fort = dict(reciprocal_rank_fusion(dense, sparse, k=60, w_dense=0.9, w_sparse=0.1))

        ratio_egal = fused_egal["y"] / fused_egal["z"]
        ratio_dense_fort = fused_dense_fort["y"] / fused_dense_fort["z"]
        assert ratio_dense_fort > ratio_egal

    def test_poids_par_defaut_identiques_a_defaut_module(self):
        """Les poids par défaut du module (DEFAULT_W_DENSE/DEFAULT_W_SPARSE)
        reproduisent le comportement non pondéré d'origine (1.0/1.0)."""
        from ai_engine.rag.evaluate_retrieval import DEFAULT_W_DENSE, DEFAULT_W_SPARSE
        assert DEFAULT_W_DENSE == 1.0
        assert DEFAULT_W_SPARSE == 1.0


class TestRecallEtPrecision:
    """Calcul recall@3 et precision@1 sur des cas simulés."""

    def test_uuid_verite_terrain_position_2_recall_1_precision_0(self, monkeypatch):
        """uuid vérité terrain en position 2 du top-3 → recall@3=1, precision@1=0."""
        ground_truth_id = build_point_id(42, "qr_technique")

        fake_top_results = [
            ("autre_id_1", 0.05),
            (ground_truth_id, 0.04),
            ("autre_id_3", 0.03),
        ]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search",
            lambda *args, **kwargs: fake_top_results,
        )

        question = {
            "id_test": "gd_test_001",
            "id_reference": 42,
            "source_type": "qr_technique",
            "categorie": "Test",
            "interface_ou_serveur": "Interface",
            "question_client": "Une question test ?",
        }

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        result = evaluate_question(
            question, mock_model, bm25=MagicMock(), vocab={}, client=MagicMock(), collection_name="test"
        )

        assert result["recall_at_3"] == 1
        assert result["trouve_top3"] is True
        assert result["rang_trouve"] == 2
        assert result["precision_at_1"] == 0

    def test_uuid_verite_terrain_absent_top3_recall_0(self, monkeypatch):
        """uuid vérité terrain absent du top-3 → recall@3=0."""
        ground_truth_id = build_point_id(99, "qr_technique")

        fake_top_results = [
            ("autre_id_1", 0.05),
            ("autre_id_2", 0.04),
            ("autre_id_3", 0.03),
        ]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search",
            lambda *args, **kwargs: fake_top_results,
        )

        question = {
            "id_test": "gd_test_002",
            "id_reference": 99,
            "source_type": "qr_technique",
            "categorie": "Test",
            "interface_ou_serveur": "Serveur",
            "question_client": "Une autre question ?",
        }

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        result = evaluate_question(
            question, mock_model, bm25=MagicMock(), vocab={}, client=MagicMock(), collection_name="test"
        )

        assert result["recall_at_3"] == 0
        assert result["trouve_top3"] is False
        assert result["rang_trouve"] is None
        assert result["precision_at_1"] == 0

    def test_uuid_verite_terrain_position_1_precision_1(self, monkeypatch):
        """uuid vérité terrain en 1ère position → precision@1=1 et recall@3=1."""
        ground_truth_id = build_point_id(7, "qr_technique")

        fake_top_results = [
            (ground_truth_id, 0.09),
            ("autre_id_2", 0.04),
            ("autre_id_3", 0.03),
        ]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search",
            lambda *args, **kwargs: fake_top_results,
        )

        question = {
            "id_test": "gd_test_003",
            "id_reference": 7,
            "source_type": "qr_technique",
            "categorie": "Test",
            "interface_ou_serveur": "Interface",
            "question_client": "Encore une question ?",
        }

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        result = evaluate_question(
            question, mock_model, bm25=MagicMock(), vocab={}, client=MagicMock(), collection_name="test"
        )

        assert result["recall_at_3"] == 1
        assert result["precision_at_1"] == 1
        assert result["rang_trouve"] == 1


class TestCoherenceUUID:
    """Non-régression : l'UUID recalculé doit être identique à celui de l'ingestion."""

    def test_uuid_identique_a_ingest_qr_technique(self):
        """
        build_point_id est importé directement depuis ingest_qr_technique.py
        (pas dupliqué) — ce test garantit qu'un futur refactor qui casserait
        cet import serait détecté, et documente la garantie de cohérence.
        """
        id_reference = 286
        source_type = "qr_technique"

        uuid_eval = build_point_id(id_reference, source_type)
        uuid_ingestion = build_point_id(id_reference, source_type)

        assert uuid_eval == uuid_ingestion

    def test_uuid_deterministe_reproductible(self):
        uuid_1 = build_point_id(410, "qr_technique")
        uuid_2 = build_point_id(410, "qr_technique")
        assert uuid_1 == uuid_2

    def test_uuid_differe_selon_id_reference(self):
        uuid_a = build_point_id(1, "qr_technique")
        uuid_b = build_point_id(2, "qr_technique")
        assert uuid_a != uuid_b


class TestRapportJSON:
    """Le rapport JSON de sortie contient tous les champs attendus par question."""

    def test_champs_attendus_dans_resultat_question(self, monkeypatch):
        ground_truth_id = build_point_id(50, "qr_technique")
        fake_top_results = [(ground_truth_id, 0.09), ("x", 0.05), ("y", 0.03)]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search",
            lambda *args, **kwargs: fake_top_results,
        )

        question = {
            "id_test": "gd_test_004",
            "id_reference": 50,
            "source_type": "qr_technique",
            "categorie": "Facturation",
            "interface_ou_serveur": "Serveur",
            "question_client": "Question ?",
        }

        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2]

        result = evaluate_question(
            question, mock_model, bm25=MagicMock(), vocab={}, client=MagicMock(), collection_name="test"
        )

        expected_keys = {
            "id_test",
            "categorie",
            "interface_ou_serveur",
            "reference_invalide",
            "trouve_top3",
            "rang_trouve",
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "recall_at_10",
            "precision_at_1",
            "reciprocal_rank",
            "score_du_premier_resultat",
            "ground_truth_id",
            "top_ids",
        }
        assert set(result.keys()) == expected_keys
        assert result["id_test"] == "gd_test_004"
        assert result["categorie"] == "Facturation"
        assert result["interface_ou_serveur"] == "Serveur"

    def test_run_produit_rapport_avec_moyennes_globales(self, monkeypatch, tmp_path):
        golden_data = {
            "questions": [
                {
                    "id_test": "gd_001",
                    "id_reference": 1,
                    "source_type": "qr_technique",
                    "categorie": "Test",
                    "interface_ou_serveur": "Interface",
                    "question_client": "Q1 ?",
                },
                {
                    "id_test": "gd_002",
                    "id_reference": 2,
                    "source_type": "qr_technique",
                    "categorie": "Test",
                    "interface_ou_serveur": "Serveur",
                    "question_client": "Q2 ?",
                },
            ]
        }
        golden_path = tmp_path / "golden.json"
        import json
        golden_path.write_text(json.dumps(golden_data), encoding="utf-8")

        report_path = tmp_path / "report.json"

        ingestion_config = {
            "chemin_source": "fake.json",
            "collection_name": "connaissances_modules",
            "module": "facturation",
            "source_type": "qr_technique",
            "modele_embedding": "paraphrase-multilingual-mpnet-base-v2",
            "longueur_troncature_reponse": 400,
            "taille_batch": 50,
        }

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.build_reference_corpus",
            lambda path: (["texte 1", "texte 2"], ingestion_config),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.build_bm25_index",
            lambda texts: (MagicMock(), {}),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.SentenceTransformer",
            lambda name: MagicMock(encode=MagicMock(return_value=MagicMock(tolist=lambda: [0.1, 0.2]))),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.QdrantClient",
            lambda url, timeout: MagicMock(),
        )

        class FakeSettings:
            qdrant_url = "http://localhost:6333"

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.get_settings", lambda: FakeSettings()
        )

        # Une question trouvée en tête, l'autre absente
        ground_truth_1 = build_point_id(1, "qr_technique")

        def fake_hybrid_search(client, collection_name, dense_query, sparse_query, limit, **kwargs):
            # Simule : la 1ère question trouve son ground truth, la 2e non
            if dense_query == [0.1, 0.2]:
                return [(ground_truth_1, 0.1), ("autre", 0.05), ("autre2", 0.03)]
            return []

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search", fake_hybrid_search
        )

        report = run(str(golden_path), "fake_ingestion_config.json", str(report_path))

        assert report["nb_questions"] == 2
        assert "recall_at_3_moyen" in report
        assert "precision_at_1_moyen" in report
        assert len(report["details"]) == 2
        assert report_path.exists()

        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        assert saved_report["nb_questions"] == 2


class TestGardeFouReferenceInvalide:
    """Garde-fou vérité-terrain (B2) : un id_reference introuvable dans Qdrant
    est exclu du calcul de recall, jamais compté comme un échec de retrieval."""

    def test_reference_introuvable_exclue_pas_comptee_comme_echec(self, monkeypatch):
        hybrid_search_appele = False

        def fake_hybrid_search(*args, **kwargs):
            nonlocal hybrid_search_appele
            hybrid_search_appele = True
            return []

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search", fake_hybrid_search
        )

        question = {
            "id_test": "gd_test_005",
            "id_reference": 999999,
            "source_type": "qr_technique",
            "categorie": "Test",
            "interface_ou_serveur": "Interface",
            "question_client": "Question dont la référence n'existe pas ?",
        }

        mock_client = MagicMock()
        mock_client.retrieve.return_value = []  # simule une référence absente de Qdrant

        result = evaluate_question(
            question, model=MagicMock(), bm25=MagicMock(), vocab={},
            client=mock_client, collection_name="test",
        )

        assert result == {
            "id_test": "gd_test_005",
            "categorie": "Test",
            "interface_ou_serveur": "Interface",
            "reference_invalide": True,
        }
        # pas de recherche inutile si la référence n'existe même pas
        assert hybrid_search_appele is False

    def test_reference_invalide_nimpacte_pas_le_recall_global(self, monkeypatch, tmp_path):
        """3 questions : 1 trouvée, 1 ratée (référence valide), 1 référence
        invalide. Le recall global doit être calculé sur 2 questions scorées
        (0.5), pas sur 3 (ce qui donnerait 0.33 si l'exclue comptait comme échec)."""
        golden_data = {
            "questions": [
                {
                    "id_test": "gd_001", "id_reference": 1, "source_type": "qr_technique",
                    "categorie": "Test", "interface_ou_serveur": "Interface", "question_client": "Q1 ?",
                },
                {
                    "id_test": "gd_002", "id_reference": 2, "source_type": "qr_technique",
                    "categorie": "Test", "interface_ou_serveur": "Serveur", "question_client": "Q2 ?",
                },
                {
                    "id_test": "gd_003", "id_reference": 999999, "source_type": "qr_technique",
                    "categorie": "Test", "interface_ou_serveur": "Interface", "question_client": "Q3 ?",
                },
            ]
        }
        golden_path = tmp_path / "golden.json"
        import json
        golden_path.write_text(json.dumps(golden_data), encoding="utf-8")
        report_path = tmp_path / "report.json"

        ingestion_config = {
            "chemin_source": "fake.json",
            "collection_name": "connaissances_modules",
            "module": "facturation",
            "source_type": "qr_technique",
            "modele_embedding": "paraphrase-multilingual-mpnet-base-v2",
            "longueur_troncature_reponse": 400,
            "taille_batch": 50,
        }

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.build_reference_corpus",
            lambda path: (["texte 1", "texte 2"], ingestion_config),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.build_bm25_index",
            lambda texts: (MagicMock(), {}),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.SentenceTransformer",
            lambda name: MagicMock(encode=MagicMock(return_value=MagicMock(tolist=lambda: [0.1, 0.2]))),
        )

        ground_truth_3_invalide = build_point_id(999999, "qr_technique")

        mock_client = MagicMock()

        def fake_retrieve(collection_name, ids, with_payload=False):
            # Seule la référence de la question 3 (999999) n'existe pas dans Qdrant.
            if ground_truth_3_invalide in ids:
                return []
            return [MagicMock()]

        mock_client.retrieve.side_effect = fake_retrieve

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.QdrantClient",
            lambda url, timeout: mock_client,
        )

        class FakeSettings:
            qdrant_url = "http://localhost:6333"

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.get_settings", lambda: FakeSettings()
        )

        ground_truth_1 = build_point_id(1, "qr_technique")

        def fake_hybrid_search(client, collection_name, dense_query, sparse_query, limit, **kwargs):
            # Résultat fixe : Q1 (ground_truth_1) trouve son propre id, Q2 non.
            return [(ground_truth_1, 0.1), ("autre", 0.05), ("autre2", 0.03)]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.hybrid_search", fake_hybrid_search
        )

        report = run(str(golden_path), "fake_ingestion_config.json", str(report_path))

        assert report["nb_questions"] == 3
        assert report["nb_questions_scorees"] == 2
        assert report["nb_questions_exclues"] == 1
        assert report["recall_at_3_moyen"] == 0.5


class TestEvaluateQuestionRerank:
    """evaluate_question_rerank() : réutilise fetch_dense_sparse_candidates +
    reciprocal_rank_fusion (pas de duplication), puis rerank cross-encoder du pool."""

    def _make_question(self, id_reference=42, id_test="gd_test_rerank"):
        return {
            "id_test": id_test,
            "id_reference": id_reference,
            "source_type": "qr_technique",
            "categorie": "Devis",
            "interface_ou_serveur": "Interface",
            "question_client": "Une question test ?",
        }

    def _make_model(self):
        mock_model = MagicMock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2]
        return mock_model

    def test_reference_invalide_exclue_sans_appel_pool(self, monkeypatch):
        appele = False

        def fake_fetch(*args, **kwargs):
            nonlocal appele
            appele = True
            return [], []

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.fetch_dense_sparse_candidates", fake_fetch
        )

        mock_client = MagicMock()
        mock_client.retrieve.return_value = []

        result = evaluate_question_rerank(
            self._make_question(), self._make_model(), bm25=MagicMock(), vocab={},
            client=mock_client, collection_name="test", reranker=MagicMock(),
        )

        assert result == {
            "id_test": "gd_test_rerank",
            "categorie": "Devis",
            "reference_invalide": True,
        }
        assert appele is False

    def test_pool_reordonne_par_le_reranker_rang_1(self, monkeypatch):
        """La bonne fiche est rang 2 avant rerank, remontée en rang 1 après."""
        ground_truth_id = build_point_id(42, "qr_technique")
        pool_ids = ["autre_1", ground_truth_id, "autre_2"]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.fetch_dense_sparse_candidates",
            lambda *args, **kwargs: ([], []),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.reciprocal_rank_fusion",
            lambda *args, **kwargs: [(pid, 1.0) for pid in pool_ids],
        )

        payloads = {
            "autre_1": {"categorie": "Autre", "question": "Q1", "reponse": "R1"},
            ground_truth_id: {"categorie": "Devis", "question": "Q2", "reponse": "R2"},
            "autre_2": {"categorie": "Autre", "question": "Q3", "reponse": "R3"},
        }

        def fake_retrieve(collection_name, ids, with_payload=False):
            existants = [MagicMock(id=ground_truth_id)] if ground_truth_id in ids else []
            if with_payload:
                return [MagicMock(id=pid, payload=payloads[pid]) for pid in ids if pid in payloads]
            return existants

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve

        # Le reranker remonte la bonne fiche (ground_truth_id) en tête
        mock_reranker = MagicMock()

        def fake_rerank(query, candidates, top_n):
            ordre = [ground_truth_id, "autre_1", "autre_2"]
            scores = {ground_truth_id: 0.95, "autre_1": 0.4, "autre_2": 0.2}
            reordonnes = sorted(candidates, key=lambda c: ordre.index(c["id"]))
            return [{**c, "reranker_score": scores[c["id"]]} for c in reordonnes][:top_n]

        mock_reranker.rerank.side_effect = fake_rerank

        result = evaluate_question_rerank(
            self._make_question(), self._make_model(), bm25=MagicMock(), vocab={},
            client=mock_client, collection_name="test", reranker=mock_reranker,
            first_stage_k=20,
        )

        assert result["in_pool"] is True
        assert result["rang_avant"] == 2
        assert result["rang_apres"] == 1
        assert result["reranker_score_bonne_fiche"] == 0.95
        assert result["top1_reranker_score"] == 0.95
        assert result["recall_at_1"] == 1
        assert result["precision_at_1"] == 1
        assert result["reciprocal_rank"] == 1.0

    def test_reference_absente_du_pool(self, monkeypatch):
        ground_truth_id = build_point_id(42, "qr_technique")
        pool_ids = ["autre_1", "autre_2"]

        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.fetch_dense_sparse_candidates",
            lambda *args, **kwargs: ([], []),
        )
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.reciprocal_rank_fusion",
            lambda *args, **kwargs: [(pid, 1.0) for pid in pool_ids],
        )

        payloads = {
            "autre_1": {"categorie": "Autre", "question": "Q1", "reponse": "R1"},
            "autre_2": {"categorie": "Autre", "question": "Q3", "reponse": "R3"},
        }

        def fake_retrieve(collection_name, ids, with_payload=False):
            if with_payload:
                return [MagicMock(id=pid, payload=payloads[pid]) for pid in ids if pid in payloads]
            return [MagicMock(id=ground_truth_id)]

        mock_client = MagicMock()
        mock_client.retrieve.side_effect = fake_retrieve

        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = lambda query, candidates, top_n: [
            {**c, "reranker_score": 0.5} for c in candidates
        ][:top_n]

        result = evaluate_question_rerank(
            self._make_question(), self._make_model(), bm25=MagicMock(), vocab={},
            client=mock_client, collection_name="test", reranker=mock_reranker,
            first_stage_k=20,
        )

        assert result["in_pool"] is False
        assert result["rang_avant"] is None
        assert result["rang_apres"] is None
        assert result["reranker_score_bonne_fiche"] is None
        assert result["recall_at_1"] == 0
        assert result["recall_at_3"] == 0
        assert result["reciprocal_rank"] == 0.0


class TestResolveGoldenDataset:
    """resolve_golden_dataset() : sélection automatique par version de fichier,
    ou override manuel (--dataset-path). Filesystem mocké (os.listdir)."""

    def test_selectionne_v2_quand_v1_et_v2_presents(self, monkeypatch):
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.os.listdir",
            lambda d: ["golden_dataset_v1.json", "golden_dataset_v2.json", "conftest.py"],
        )
        assert resolve_golden_dataset(search_dir="tests") == "tests/golden_dataset_v2.json"

    def test_override_path_respecte(self, monkeypatch):
        appele = False

        def fake_listdir(d):
            nonlocal appele
            appele = True
            return ["golden_dataset_v1.json", "golden_dataset_v2.json"]

        monkeypatch.setattr("ai_engine.rag.evaluate_retrieval.os.listdir", fake_listdir)

        resultat = resolve_golden_dataset(override_path="tests/golden_dataset_custom.json")

        assert resultat == "tests/golden_dataset_custom.json"
        assert appele is False

    def test_fichier_sans_suffixe_traite_comme_v0(self, monkeypatch):
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.os.listdir",
            lambda d: ["golden_dataset.json"],
        )
        assert resolve_golden_dataset(search_dir="tests") == "tests/golden_dataset.json"

    def test_v0_sans_suffixe_a_priorite_minimale_face_a_v1(self, monkeypatch):
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.os.listdir",
            lambda d: ["golden_dataset.json", "golden_dataset_v1.json"],
        )
        assert resolve_golden_dataset(search_dir="tests") == "tests/golden_dataset_v1.json"

    def test_aucun_fichier_correspondant_leve_erreur(self, monkeypatch):
        monkeypatch.setattr(
            "ai_engine.rag.evaluate_retrieval.os.listdir",
            lambda d: ["autre_fichier.json"],
        )
        with pytest.raises(FileNotFoundError):
            resolve_golden_dataset(search_dir="tests")
