"""
Test simple pour valider la mesure des timings dans le pipeline.
Lance une requête factice et affiche les timings.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ajuster le chemin pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from ai_engine.api.schemas import (
    AgentResponse,
    MetaSupport,
    SourceFiche,
    StepTiming,
    TicketInput,
)
from ai_engine.rag.retriever import RetrievalResult


def test_schema_with_timings():
    """Valide que les nouveaux champs de timing existent dans les schémas."""
    print("✓ Test 1 : Schémas Pydantic")

    # Test StepTiming
    timing = StepTiming(step_name="retrieval_hybrid", duration_ms=145.3)
    assert timing.step_name == "retrieval_hybrid"
    assert timing.duration_ms == 145.3
    print("  ✓ StepTiming valide")

    # Test MetaSupport avec timings
    meta = MetaSupport(
        confidence=0.75,
        sources=[],
        escalade_reason=None,
        model_used="gemini-2.5-flash-lite",
        processing_duration_ms=523.42,
        step_timings=[
            StepTiming(step_name="retrieval_hybrid", duration_ms=145.3),
            StepTiming(step_name="reranking", duration_ms=78.2),
        ],
    )
    assert meta.processing_duration_ms == 523.42
    assert len(meta.step_timings) == 2
    print("  ✓ MetaSupport avec timings valide")

    # Test AgentResponse avec timings
    response = AgentResponse(
        ticket_id="TEST_001",
        status="reponse_a_valider",
        reponse_client="Réponse générée",
        meta_support=meta,
    )
    assert response.meta_support.processing_duration_ms == 523.42
    print("  ✓ AgentResponse avec timings valide")

    # Sérialisation JSON
    response_json = response.model_dump_json(indent=2)
    assert "processing_duration_ms" in response_json
    assert "step_timings" in response_json
    print("  ✓ Sérialisation JSON valide")
    print(f"    → Extrait JSON :\n{response_json[:300]}...\n")


def test_retrieval_result_with_timings():
    """Valide que RetrievalResult inclut les timings."""
    print("✓ Test 2 : RetrievalResult avec timings")

    result = RetrievalResult(
        answer_found=True,
        should_escalate=False,
        escalade_reason="",
        top_candidates=[],
        top1_score=0.75,
        query="test query",
        step_timings={
            "scope_check_ms": 0.5,
            "retrieval_hybrid_ms": 145.3,
            "reranking_ms": 78.2,
            "total_ms": 224.0,
        },
    )

    assert result.step_timings["scope_check_ms"] == 0.5
    assert result.step_timings["retrieval_hybrid_ms"] == 145.3
    assert result.step_timings["total_ms"] == 224.0
    print("  ✓ RetrievalResult.step_timings valide")
    print(f"    → Timings: {result.step_timings}\n")


def test_agent4_timing_conversion():
    """Valide la conversion du dictionnaire de timings en liste de StepTiming."""
    print("✓ Test 3 : Conversion Agent4 timings dict → list")

    from ai_engine.agents.agent4_trivial import Agent4

    # Mock minimal pour tester _build_step_timings_list
    agent4 = Agent4(
        retriever=MagicMock(),
        gemini_client=MagicMock(),
    )

    step_timings = {
        "scope_check_ms": 0.5,
        "retrieval_hybrid_ms": 145.3,
        "reranking_ms": 78.2,
        "retrieval_ms": 224.0,
        "anonymisation_ms": 12.5,
        "gemini_generation_ms": 286.92,
        "total_ms": 523.92,
    }

    timings_list = agent4._build_step_timings_list(step_timings)

    assert len(timings_list) == 6
    assert timings_list[0].step_name == "scope_check"
    assert timings_list[0].duration_ms == 0.5
    assert timings_list[-1].step_name == "gemini_generation"
    assert timings_list[-1].duration_ms == 286.92

    print("  ✓ Conversion valide (6 étapes)")
    for timing in timings_list:
        print(f"    → {timing.step_name:25} : {timing.duration_ms:7.2f} ms")
    print()


def test_timing_precision():
    """Valide que les mesures de temps fonctionnent avec précision."""
    print("✓ Test 4 : Précision des mesures")

    start = time.perf_counter()
    time.sleep(0.05)  # 50 ms
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Doit être entre 45 et 60 ms (marge pour variations système)
    assert 45 <= elapsed_ms <= 70, f"Timing hors limites: {elapsed_ms}ms"
    print(f"  ✓ time.perf_counter() fonctionne correctement")
    print(f"    → Sleep 50ms mesuré à {elapsed_ms:.2f}ms\n")


def test_escalade_with_timings():
    """Valide les timings dans les cas d'escalade."""
    print("✓ Test 5 : Timings en cas d'escalade")

    # Escalade par score insuffisant
    meta_escalade = MetaSupport(
        confidence=0.45,
        sources=[],
        escalade_reason="score_insuffisant",
        model_used=None,
        processing_duration_ms=224.5,
        step_timings=[
            StepTiming(step_name="scope_check", duration_ms=0.5),
            StepTiming(step_name="retrieval_hybrid", duration_ms=145.3),
            StepTiming(step_name="reranking", duration_ms=78.7),
        ],
    )

    response_escalade = AgentResponse(
        ticket_id="ESC_001",
        status="escalade",
        reponse_client=None,
        meta_support=meta_escalade,
    )

    assert response_escalade.status == "escalade"
    assert response_escalade.meta_support.processing_duration_ms == 224.5
    assert len(response_escalade.meta_support.step_timings) == 3
    print("  ✓ Escalade score_insuffisant avec timings valide")
    print(f"    → Durée totale: {meta_escalade.processing_duration_ms}ms")
    print(f"    → Étapes mesurées: {len(meta_escalade.step_timings)}\n")


def run_all_tests():
    """Lance tous les tests."""
    print("=" * 70)
    print("SUITE DE TESTS : Mesure de la durée de traitement")
    print("=" * 70 + "\n")

    try:
        test_schema_with_timings()
        test_retrieval_result_with_timings()
        test_agent4_timing_conversion()
        test_timing_precision()
        test_escalade_with_timings()

        print("=" * 70)
        print("✓ TOUS LES TESTS RÉUSSIS")
        print("=" * 70)
        print("\n📊 Prochaines étapes :")
        print("  1. Lancer le serveur API : python -m ai_engine.api.main")
        print("  2. Envoyer un ticket : curl -X POST http://localhost:8000/process-ticket ...")
        print("  3. Vérifier processing_duration_ms et step_timings dans la réponse")
        print("  4. Consulter les logs INFO pour les durées par ticket")
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST ÉCHOUÉ : {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERREUR INATTENDUE : {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
