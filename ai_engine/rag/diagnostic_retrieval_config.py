"""
ai_engine/rag/diagnostic_retrieval_config.py

Script de diagnostic en lecture seule qui:
1. Extrait les constantes de configuration du pipeline RAG
2. Appelle les fonctions de retrieval existantes avec journalisation
3. Rapporte rangs et scores par etage
4. Ne modifie JAMAIS le pipeline - instrumentation par lecture seule

Constantes de configuration tracees:
- FETCH_LIMIT (evaluation) : nb resultats remontes par dense/sparse AVANT fusion RRF
- rag_first_stage_k (production) : taille du pool APRES fusion RRF avant reranking
- TOP_K_MAX (evaluation) : profondeur d'evaluation unique sur une seule liste
- top_n (production) : nb resultats retournes apres reranking
- RRF_K : constante RRF standard
- rag_confidence_threshold : seuil du gate d'abstention

Usage :
    python -m ai_engine.rag.diagnostic_retrieval_config "ma question" \\
        --module Facturation --priorite Assistance

ou avec verite-terrain (id_reference de golden_dataset.json) :
    python -m ai_engine.rag.diagnostic_retrieval_config "question test" \\
        --id-reference 286 --source-type qr_technique
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Importe les constantes d'evaluation (read-only)
from ai_engine.rag.evaluate_retrieval import (
    RRF_K,
    FETCH_LIMIT,
    TOP_K_MAX,
    RECALL_DEPTHS,
)

# Importe les constantes de production
from ai_engine.rag.retriever import Retriever
from ai_engine.config import get_settings


def _extract_config_constants() -> dict[str, Any]:
    """
    Extrait TOUTES les constantes de configuration du pipeline RAG.
    Lecture seule - aucune modification.
    """
    settings = get_settings()

    return {
        "SECTION_EVALUATION": {
            "FETCH_LIMIT": FETCH_LIMIT,
            "TOP_K_MAX": TOP_K_MAX,
            "RECALL_DEPTHS": RECALL_DEPTHS,
            "RRF_K": RRF_K,
            "description_FETCH_LIMIT": (
                "Nb resultats remontes par dense/sparse AVANT fusion RRF "
                "(evaluation, evaluate_retrieval.py)"
            ),
            "description_TOP_K_MAX": (
                "Profondeur d'evaluation unique remontee par question "
                "(correspond a max(RECALL_DEPTHS))"
            ),
            "description_RRF_K": (
                "Constante RRF standard (Cormack et al. 2009) : "
                "score = somme de 1/(k+rang)"
            ),
        },
        "SECTION_PRODUCTION": {
            "rag_first_stage_k": settings.rag_first_stage_k,
            "rag_confidence_threshold": settings.rag_confidence_threshold,
            "rag_module_filter_enabled": settings.rag_module_filter_enabled,
            "rag_scope_modules": settings.rag_scope_modules,
            "rag_scope_priorities": settings.rag_scope_priorities,
            "description_rag_first_stage_k": (
                "Taille du pool hybride (dense+sparse) remont avant reranking "
                "(production, retriever.py)"
            ),
            "description_rag_confidence_threshold": (
                "Seuil du gate d'abstention : top1_reranker_score minimum "
                "pour repondre automatiquement"
            ),
        },
        "NOTE_AMBIGUITES": {
            "key_retrieval_k": (
                f"FETCH_LIMIT (evaluation) = {FETCH_LIMIT} vs "
                f"rag_first_stage_k (production) = {settings.rag_first_stage_k}"
            ),
            "top_k_values": (
                f"TOP_K_MAX (evaluation) = {TOP_K_MAX} vs "
                f"top_n (production) = parametre de retrieve(), defaut 3"
            ),
            "note": (
                "Les deux valeurs FETCH_LIMIT et rag_first_stage_k sont egales par design "
                "(20), mais peuvent diverger - l'evaluation et la production utilisent "
                "les memes fonctions (evaluate_retrieval.fetch_dense_sparse_candidates). "
                "TOP_K_MAX est utilise uniquement pour recall@1/@3/@5/@10 en evaluation ; "
                "top_n est le parametre final de retrieve()."
            ),
        },
    }


def _print_config_report(config: dict[str, Any]) -> None:
    """Affiche le rapport des constantes de configuration."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CONFIGURATION - CONSTANTES DU PIPELINE RAG")
    print("=" * 80)

    print("\n---- SECTION EVALUATION (evaluate_retrieval.py) ----")
    eval_section = config["SECTION_EVALUATION"]
    for key in ["FETCH_LIMIT", "TOP_K_MAX", "RRF_K"]:
        val = eval_section[key]
        desc = eval_section.get(f"description_{key}", "")
        print(f"\n  {key} = {val}")
        if desc:
            print(f"      {desc}")

    print("\n  RECALL_DEPTHS = ", eval_section["RECALL_DEPTHS"])
    print(f"      Profondeurs d'evaluation utilisees pour recall/precision/MRR")

    print("\n---- SECTION PRODUCTION (retriever.py, config.py) ----")
    prod_section = config["SECTION_PRODUCTION"]
    for key in ["rag_first_stage_k", "rag_confidence_threshold", "rag_module_filter_enabled"]:
        val = prod_section[key]
        desc = prod_section.get(f"description_{key}", "")
        print(f"\n  {key} = {val}")
        if desc:
            print(f"      {desc}")

    print(f"\n  rag_scope_modules = {prod_section['rag_scope_modules']}")
    print(f"  rag_scope_priorities = {prod_section['rag_scope_priorities']}")

    print("\n---- NOTES SUR LES AMBIGUITES ----")
    notes = config["NOTE_AMBIGUITES"]
    for key, val in notes.items():
        if key != "note":
            print(f"\n  {key}:")
            print(f"      {val}")
    print(f"\n  {notes['note']}")

    print("\n" + "=" * 80)


def _print_pipeline_execution_header(query: str, module: str | None, priorite: str | None) -> None:
    """Affiche l'en-tete du diagnostic d'execution."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC EXECUTION - PIPELINE RETRIEVER.RETRIEVE()")
    print("=" * 80)
    print(f"\nQuestion : {query!r}")
    if module:
        print(f"Module : {module}")
    if priorite:
        print(f"Priorite : {priorite}")
    print()


def _diagnose_retrieval(
    retriever: Retriever,
    query: str,
    module: str | None,
    priorite: str | None,
    top_n: int,
) -> None:
    """
    Appelle Retriever.retrieve() et journalise les resultats par etage.
    Instrumentation en lecture seule - n'appelle PAS debug_retriever.
    """
    print("---- ETAPE 0 : PRE-QUALIFICATION SCOPE ----")
    settings = retriever.settings

    if settings.rag_module_filter_enabled:
        scope_modules_norm = {m.strip().lower() for m in settings.rag_scope_modules}
        scope_priorities_norm = {p.strip().lower() for p in settings.rag_scope_priorities}

        if module is not None:
            module_norm = module.strip().lower()
            if module_norm not in scope_modules_norm:
                print(f"[ESCALADE] module '{module}' hors-scope.")
                print(f"Modules acceptes : {settings.rag_scope_modules}")
                return
            else:
                print(f"[OK] module '{module}' accepte (dans {settings.rag_scope_modules})")

        if priorite is not None:
            priorite_norm = priorite.strip().lower()
            if priorite_norm not in scope_priorities_norm:
                print(f"[ESCALADE] priorite '{priorite}' hors-scope.")
                print(f"Priorites acceptees : {settings.rag_scope_priorities}")
                return
            else:
                print(f"[OK] priorite '{priorite}' acceptee (dans {settings.rag_scope_priorities})")
    else:
        print("[BYPASS] Filtre de scope desactive")

    print(f"\n---- ETAPE 1-3 : RETRIEVAL HYBRIDE + RERANKING + GATE ----")
    try:
        result = retriever.retrieve(query, module=module, priorite=priorite, top_n=top_n)

        print(f"Query : {result.query!r}")
        print(f"Answer found : {result.answer_found}")
        print(f"Top1 score : {result.top1_score:.4f} (seuil : {settings.rag_confidence_threshold})")
        print(f"Should escalate : {result.should_escalate}")
        if result.escalade_reason:
            print(f"Escalade reason : {result.escalade_reason}")

        print(f"\n---- ETAPE 4 : RESULTATS APRES RERANKING (top {len(result.top_candidates)}) ----")
        if result.top_candidates:
            for rang, candidate in enumerate(result.top_candidates[:3], start=1):
                payload = candidate.get("payload", {})
                categorie = payload.get("categorie", "?")
                question = (payload.get("question") or "")[:70]
                score = candidate.get("reranker_score", 0.0)
                print(f"\n  Rang {rang} :")
                print(f"    ID : {candidate.get('id')}")
                print(f"    Score : {score:.4f}")
                print(f"    Categorie : {categorie}")
                print(f"    Question : {question}")
        else:
            print("(Aucun resultat)")

    except Exception as e:
        print(f"[ERREUR] Exception durant le retrieval : {type(e).__name__}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic en lecture seule du pipeline RAG : "
            "extraction des constantes de config + execution avec journalisation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python -m ai_engine.rag.diagnostic_retrieval_config "
            "\"comment imprimer un avoir ?\"\n"
            "  python -m ai_engine.rag.diagnostic_retrieval_config "
            "\"ma question\" --module Facturation --priorite Assistance\n"
            "  python -m ai_engine.rag.diagnostic_retrieval_config "
            "\"ma question\" --id-reference 286 --source-type qr_technique"
        ),
    )
    parser.add_argument("query", type=str, help="Question a tracer")
    parser.add_argument("--module", type=str, default=None, help="Module du ticket")
    parser.add_argument("--priorite", type=str, default=None, help="Priorite du ticket")
    parser.add_argument("--top-n", type=int, default=3, help="Nombre de resultats a retourner")
    parser.add_argument(
        "--id-reference", type=str, default=None,
        help="id_reference de la verite-terrain a localiser (optionnel)",
    )
    parser.add_argument(
        "--source-type", type=str, default="qr_technique",
        help="source_type associe a --id-reference (defaut : qr_technique)",
    )
    parser.add_argument(
        "--save-config-report", type=str, default=None,
        help=(
            "Sauvegarder le rapport de configuration en JSON "
            "(ex: diagnostic_config.json)"
        ),
    )
    args = parser.parse_args()

    # Etape 1 - Extraction des constantes (lecture seule)
    config = _extract_config_constants()
    _print_config_report(config)

    # Etape 2 - Sauvegarde optionnelle du rapport JSON
    if args.save_config_report:
        with open(args.save_config_report, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Rapport de configuration sauvegarde : {args.save_config_report}\n")

    # Etape 3 - Execution du diagnostic avec journalisation
    _print_pipeline_execution_header(args.query, args.module, args.priorite)

    retriever = Retriever()
    _diagnose_retrieval(
        retriever,
        args.query,
        module=args.module,
        priorite=args.priorite,
        top_n=args.top_n,
    )

    print("\n" + "=" * 80)
    print("FIN DU DIAGNOSTIC")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
