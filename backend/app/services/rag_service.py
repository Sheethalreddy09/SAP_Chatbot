import logging
from typing import Dict, Any, List, Generator
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.services.sap_classifier import sap_classifier
from backend.app.services.retrieval_service import retrieval_service
from backend.app.services.reranking_service import reranking_service
from backend.app.services.grounding_service import grounding_service
from backend.app.services.web_search_service import web_search_service
from backend.app.services.answer_verification_service import answer_verification_service
from backend.app.services.query_rewriter import query_rewriter
from backend.app.services.evidence_context import prepare_scoped_evidence

logger = logging.getLogger(__name__)


class RAGService:
    @staticmethod
    def _checked_response(query, checked, source_type, comparison):
        grounding = 0.0
        if checked["accepted_count"]:
            # Score only the quoted evidence used by surviving claims.
            sources = checked["citations"] if source_type == "knowledge_base" else checked["web_sources"]
            evidence = [{**item, "chunk_text": item["snippet"]} for item in sources]
            grounding = grounding_service.evaluate_grounding(
                query, checked["answer"], evidence, source_type=source_type)
        return {
            "answer": checked["answer"], "source_type": source_type,
            "verification_status": checked["verification_status"],
            "grounding_score": grounding,
            "is_in_rag_pipeline": comparison["is_in_rag_pipeline"],
            "kb_score": comparison["kb_score"], "web_score": comparison["web_score"],
            "winning_score": comparison["winning_score"], "selected_source": source_type,
            "citations": checked["citations"], "web_sources": checked["web_sources"],
        }

    def process_query(self, db: Session, query: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the full, verified RAG pipeline:
        1. Context-aware conversational follow-up resolution
        2. Strict SAP domain restriction validation (performed on resolved query)
        3. Accelerated hybrid retrieval from internal knowledge base
        4. Authoritative SAP web search retrieval
        5. Objective multi-factor reranking & score comparison (KB Score vs Web Score)
        6. Selection of highest-scoring evidence passed to Groq LLM
        7. Structured draft, separate source audit, checked-text rendering and heuristic score
        """
        # STEP 1: Conversational Context Resolution
        resolution = query_rewriter.resolve_query(query, chat_history=chat_history)
        resolved_query = resolution["resolved_query"]
        is_follow_up = resolution["is_follow_up"]
        conversation_topic = resolution["conversation_topic"]
        cleaned_query = resolved_query.strip()

        # STEP 2: Domain Restriction Validation on resolved query
        is_sap, reason = sap_classifier.classify(cleaned_query)

        logger.info(
            f"\n================ QUERY RESOLUTION ================\n"
            f"Original query: {query}\n"
            f"Detected follow-up: {is_follow_up}\n"
            f"Conversation topic: {conversation_topic}\n"
            f"Resolved query: {cleaned_query}\n"
            f"Scope: {'SAP' if is_sap else 'NON_SAP'}\n"
            f"=================================================="
        )

        if not is_sap:
            return {
                "answer": "I can only help with SAP and SAP-related topics.",
                "source_type": "refusal",
                "grounding_score": 0.0,
                "is_in_rag_pipeline": False,
                "kb_score": 0.0,
                "web_score": 0.0,
                "winning_score": 0.0,
                "selected_source": "refusal",
                "citations": [],
                "web_sources": []
            }

        # STEP 2: Retrieve from Internal Knowledge Base
        kb_candidates = retrieval_service.hybrid_search(db, cleaned_query, top_k=settings.TOP_K)

        # STEP 3: Retrieve from Authoritative SAP Web Sources if enabled
        web_results = []
        if settings.WEB_FALLBACK_ENABLED:
            try:
                web_results = web_search_service.search_sap_authoritative(cleaned_query)
            except Exception as e:
                logger.warning(f"Web search error: {e}")
                web_results = []

        # STEP 4: Objective Multi-factor Reranking & Score Comparison
        comparison = reranking_service.compare_and_select(
            cleaned_query,
            kb_candidates,
            web_results,
            threshold=settings.SIMILARITY_THRESHOLD
        )

        winner = comparison["winner"]
        is_in_rag_pipeline = comparison["is_in_rag_pipeline"]
        kb_score = comparison["kb_score"]
        web_score = comparison["web_score"]
        winning_score = comparison["winning_score"]
        top_kb_chunks = comparison["top_kb_chunks"]
        top_web_results = comparison["top_web_results"]

        logger.info(
            f"RAG Comparison for '{cleaned_query[:40]}': Winner={winner}, "
            f"KB Score={kb_score:.4f}, Web Score={web_score:.4f}, In-KB={is_in_rag_pipeline}"
        )

        if winner in {"knowledge_base", "web"}:
            evidence = (prepare_scoped_evidence(db, top_kb_chunks)
                        if winner == "knowledge_base" else top_web_results)
            if evidence:
                checked = answer_verification_service.generate(cleaned_query, evidence, winner)
                return self._checked_response(cleaned_query, checked, winner, comparison)

        # Case C: Insufficient information everywhere
        notice = (
            f"I couldn't find sufficient information about this in either the internal "
            f"SAP knowledge base (Relevance Score: {kb_score:.2f}) or authoritative SAP web documentation "
            f"(Relevance Score: {web_score:.2f})."
        )
        return {
            "answer": notice,
            "source_type": "knowledge_base",
            "grounding_score": 0.0,
            "is_in_rag_pipeline": False,
            "kb_score": kb_score,
            "web_score": web_score,
            "winning_score": winning_score,
            "selected_source": "none",
            "citations": [],
            "web_sources": []
        }

    def stream_query(self, db: Session, query: str, chat_history: List[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Executes progressive streaming through verified RAG pipeline:
        Emits status events while drafting and checking; only then releases checked text,
        and finally yields the done event with complete citations, grounding score, and source metadata.
        """
        # STAGE 1: Thinking / Conversational Context Resolution & Domain Validation
        yield {
            "type": "status",
            "stage": "thinking",
            "message": "Thinking..."
        }

        # Resolve conversational follow-up
        resolution = query_rewriter.resolve_query(query, chat_history=chat_history)
        resolved_query = resolution["resolved_query"]
        is_follow_up = resolution["is_follow_up"]
        conversation_topic = resolution["conversation_topic"]
        cleaned_query = resolved_query.strip()

        # Scope classification on resolved query
        is_sap, reason = sap_classifier.classify(cleaned_query)

        logger.info(
            f"\n================ STREAM QUERY RESOLUTION ================\n"
            f"Original query: {query}\n"
            f"Detected follow-up: {is_follow_up}\n"
            f"Conversation topic: {conversation_topic}\n"
            f"Resolved query: {cleaned_query}\n"
            f"Scope: {'SAP' if is_sap else 'NON_SAP'}\n"
            f"========================================================="
        )

        if not is_sap:
            refusal_text = "I can only help with SAP and SAP-related topics."
            yield {
                "type": "token",
                "content": refusal_text
            }
            yield {
                "type": "done",
                "answer": refusal_text,
                "source_type": "refusal",
                "grounding_score": 0.0,
                "is_in_rag_pipeline": False,
                "kb_score": 0.0,
                "web_score": 0.0,
                "winning_score": 0.0,
                "selected_source": "refusal",
                "citations": [],
                "web_sources": []
            }
            return

        # STAGE 2: Searching SAP Knowledge (Internal KB + Web)
        yield {
            "type": "status",
            "stage": "retrieval",
            "message": "Searching SAP knowledge..."
        }

        kb_candidates = retrieval_service.hybrid_search(db, cleaned_query, top_k=settings.TOP_K)

        web_results = []
        if settings.WEB_FALLBACK_ENABLED:
            try:
                web_results = web_search_service.search_sap_authoritative(cleaned_query)
            except Exception as e:
                logger.warning(f"Web search error: {e}")
                web_results = []

        # STAGE 3: Multi-factor Reranking & Evidence Analysis
        yield {
            "type": "status",
            "stage": "reranking",
            "message": "Analyzing retrieved SAP evidence..."
        }

        comparison = reranking_service.compare_and_select(
            cleaned_query,
            kb_candidates,
            web_results,
            threshold=settings.SIMILARITY_THRESHOLD
        )

        winner = comparison["winner"]
        is_in_rag_pipeline = comparison["is_in_rag_pipeline"]
        kb_score = comparison["kb_score"]
        web_score = comparison["web_score"]
        winning_score = comparison["winning_score"]
        top_kb_chunks = comparison["top_kb_chunks"]
        top_web_results = comparison["top_web_results"]

        logger.info(
            f"Streaming RAG: Winner={winner}, KB={kb_score:.3f}, Web={web_score:.3f}, In-KB={is_in_rag_pipeline}"
        )

        # STAGE 4: Preparing Grounded Answer
        yield {
            "type": "status",
            "stage": "grounding",
            "message": "Preparing grounded answer..."
        }

        if winner in {"knowledge_base", "web"}:
            evidence = (prepare_scoped_evidence(db, top_kb_chunks)
                        if winner == "knowledge_base" else top_web_results)
            if evidence:
                checked = None
                for event in answer_verification_service.stream_check(cleaned_query, evidence, winner):
                    if event["type"] == "status":
                        yield event
                    elif event["type"] == "checked":
                        checked = event["result"]
                if checked is None:
                    raise RuntimeError("Evidence check returned no result")
                result = self._checked_response(cleaned_query, checked, winner, comparison)
                # Both the persisted answer and every token come from the same checked text.
                # No draft is released even if the provider or audit fails.
                yield {"type": "token", "content": result["answer"]}
                yield {"type": "done", **result}
                return

        # Case C: Insufficient info everywhere
        notice = (
            f"I couldn't find sufficient information about this in either the internal "
            f"SAP knowledge base (Relevance Score: {kb_score:.2f}) or authoritative SAP web documentation "
            f"(Relevance Score: {web_score:.2f})."
        )
        yield {
            "type": "token",
            "content": notice
        }
        yield {
            "type": "done",
            "answer": notice,
            "source_type": "knowledge_base",
            "grounding_score": 0.0,
            "is_in_rag_pipeline": False,
            "kb_score": kb_score,
            "web_score": web_score,
            "winning_score": winning_score,
            "selected_source": "none",
            "citations": [],
            "web_sources": []
        }


rag_service = RAGService()


