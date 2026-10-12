import json
import logging
from typing import List, Dict, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from backend.app.database import get_db, SessionLocal
from backend.app.models.user import User
from backend.app.models.chat import Chat
from backend.app.models.message import Message
from backend.app.models.citation import Citation
from backend.app.models.web_source import WebSource
from backend.app.schemas.message import MessageCreate, MessageResponse
from backend.app.security.auth import get_current_user
from backend.app.services.rag_service import rag_service
from backend.app.services.llm_service import llm_service
from backend.app.services.retrieval_service import retrieval_service
from backend.app.services.reranking_service import reranking_service
from backend.app.services.grounding_service import grounding_service
from backend.app.services.web_search_service import web_search_service
from backend.app.services.sap_classifier import sap_classifier
from backend.app.config import settings

router = APIRouter(prefix="/chats/{chat_id}/messages", tags=["Messages"])

def verify_chat_ownership(chat_id: str, user_id: str, db: Session) -> Chat:
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    return chat

@router.get("", response_model=List[MessageResponse])
def get_messages(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    verify_chat_ownership(chat_id, current_user.id, db)
    messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    result = []
    for msg in messages:
        c_list = [
            {
                "id": c.id,
                "document": c.document_name,
                "page": c.page_number,
                "section": c.section,
                "score": c.similarity_score,
                "snippet": c.citation_text
            }
            for c in msg.citations
        ]
        w_list = [
            {
                "id": w.id,
                "title": w.title,
                "url": w.url,
                "domain": w.domain,
                "snippet": w.snippet
            }
            for w in msg.web_sources
        ]
        result.append(
            MessageResponse(
                id=msg.id,
                chat_id=msg.chat_id,
                role=msg.role,
                content=msg.content,
                source_type=msg.source_type,
                grounding_score=msg.grounding_score,
                created_at=msg.created_at,
                citations=c_list,
                web_sources=w_list
            )
        )
    return result

@router.post("")
def send_message(
    chat_id: str,
    msg_in: MessageCreate,
    stream: bool = Query(False, description="Enable Server-Sent Events (SSE) streaming"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = verify_chat_ownership(chat_id, current_user.id, db)

    # 1. Save user message
    user_msg = Message(
        chat_id=chat_id,
        role="user",
        content=msg_in.content.strip()
    )
    db.add(user_msg)
    
    # Auto-generate chat title from first question if default
    if chat.title == "New SAP Chat" or not chat.title:
        words = msg_in.content.strip().split()
        chat.title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)

    # Fetch recent conversation history (excluding the new user_msg)
    prev_messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .filter(Message.id != user_msg.id)
        .order_by(Message.created_at.asc())
        .all()
    )
    chat_history = [
        {"role": m.role, "content": m.content, "source_type": m.source_type}
        for m in prev_messages
    ]

    # 2. Check SSE streaming flow
    if stream:
        return handle_streaming_response(chat_id, msg_in.content.strip(), chat_history=chat_history)

    # 3. Synchronous RAG flow
    rag_result = rag_service.process_query(db, msg_in.content.strip(), chat_history=chat_history)

    # 4. Save assistant message
    assistant_msg = Message(
        chat_id=chat_id,
        role="assistant",
        content=rag_result["answer"],
        source_type=rag_result["source_type"],
        grounding_score=rag_result["grounding_score"]
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    # 5. Persist citations or web sources
    if rag_result["citations"]:
        for c in rag_result["citations"]:
            cit = Citation(
                message_id=assistant_msg.id,
                source_type="knowledge_base",
                document_id="doc-" + c["document"],
                document_name=c["document"],
                chunk_id="chunk-ref",
                page_number=c["page"],
                section=c["section"],
                similarity_score=c["score"],
                citation_text=c["snippet"]
            )
            db.add(cit)
        db.commit()

    if rag_result["web_sources"]:
        for w in rag_result["web_sources"]:
            ws = WebSource(
                message_id=assistant_msg.id,
                url=w["url"],
                title=w["title"],
                domain=w["domain"],
                snippet=w["snippet"]
            )
            db.add(ws)
        db.commit()

    db.refresh(assistant_msg)

    return MessageResponse(
        id=assistant_msg.id,
        chat_id=assistant_msg.chat_id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        source_type=assistant_msg.source_type,
        grounding_score=assistant_msg.grounding_score,
        is_in_rag_pipeline=rag_result.get("is_in_rag_pipeline"),
        kb_score=rag_result.get("kb_score"),
        web_score=rag_result.get("web_score"),
        winning_score=rag_result.get("winning_score"),
        created_at=assistant_msg.created_at,
        citations=[
            {
                "id": c.id,
                "document": c.document_name,
                "page": c.page_number,
                "section": c.section,
                "score": c.similarity_score,
                "snippet": c.citation_text
            }
            for c in assistant_msg.citations
        ],
        web_sources=[
            {
                "id": w.id,
                "title": w.title,
                "url": w.url,
                "domain": w.domain,
                "snippet": w.snippet
            }
            for w in assistant_msg.web_sources
        ]
    )


@router.post("/stream")
def send_message_stream(
    chat_id: str,
    msg_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Direct POST /chats/{chat_id}/messages/stream endpoint for SSE streaming."""
    chat = verify_chat_ownership(chat_id, current_user.id, db)

    # 1. Save user message
    user_msg = Message(
        chat_id=chat_id,
        role="user",
        content=msg_in.content.strip()
    )
    db.add(user_msg)

    if chat.title == "New SAP Chat" or not chat.title:
        words = msg_in.content.strip().split()
        chat.title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")

    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user_msg)

    prev_messages = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .filter(Message.id != user_msg.id)
        .order_by(Message.created_at.asc())
        .all()
    )
    chat_history = [
        {"role": m.role, "content": m.content, "source_type": m.source_type}
        for m in prev_messages
    ]

    return handle_streaming_response(chat_id, msg_in.content.strip(), chat_history=chat_history)


@router.post("/save-partial")
def save_partial_message(
    chat_id: str,
    msg_in: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Explicitly saves a partial assistant message when generation is stopped."""
    chat = verify_chat_ownership(chat_id, current_user.id, db)
    partial_msg = Message(
        chat_id=chat_id,
        role="assistant",
        content=msg_in.content.strip(),
        source_type="interrupted",
        grounding_score=0.0
    )
    db.add(partial_msg)
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(partial_msg)
    return {"status": "saved", "message_id": partial_msg.id}


def handle_streaming_response(chat_id: str, query: str, db: Session = None, chat_history: List[Dict[str, Any]] = None):
    """
    Generates Server-Sent Events (SSE) stream using the complete verified RAG pipeline.
    Progressively yields structured status stages, LLM token stream, and done event with citations/grounding.
    Persists the completed assistant message in the database once generation concludes.
    """
    def event_stream():
        save_db = SessionLocal()
        accumulated_text = ""
        done_payload = None

        hist = chat_history
        if hist is None:
            prev_msgs = (
                save_db.query(Message)
                .filter(Message.chat_id == chat_id)
                .order_by(Message.created_at.asc())
                .all()
            )
            hist = []
            for m in prev_msgs:
                if m.role == "user" and m.content.strip() == query.strip():
                    continue
                hist.append({"role": m.role, "content": m.content, "source_type": m.source_type})

        try:
            for event in rag_service.stream_query(save_db, query, chat_history=hist):
                event_type = event.get("type")

                if event_type == "token":
                    accumulated_text += event.get("content", "")

                elif event_type == "done":
                    done_payload = event
                    full_answer = event.get("answer", accumulated_text)
                    source_type = event.get("source_type", "knowledge_base")
                    grounding = event.get("grounding_score", 0.0)
                    citations = event.get("citations", [])
                    web_sources = event.get("web_sources", [])

                    # Persist single assistant message to DB
                    assistant_msg = Message(
                        chat_id=chat_id,
                        role="assistant",
                        content=full_answer,
                        source_type=source_type,
                        grounding_score=grounding
                    )
                    save_db.add(assistant_msg)
                    save_db.commit()
                    save_db.refresh(assistant_msg)

                    if citations:
                        for c in citations:
                            save_db.add(Citation(
                                message_id=assistant_msg.id,
                                source_type="knowledge_base",
                                document_id="doc-" + str(c.get("document", "")),
                                document_name=c.get("document", ""),
                                chunk_id="chunk-ref",
                                page_number=c.get("page"),
                                section=c.get("section"),
                                similarity_score=c.get("score"),
                                citation_text=c.get("snippet")
                            ))
                        save_db.commit()

                    if web_sources:
                        for w in web_sources:
                            save_db.add(WebSource(
                                message_id=assistant_msg.id,
                                url=w.get("url"),
                                title=w.get("title"),
                                domain=w.get("domain"),
                                snippet=w.get("snippet")
                            ))
                        save_db.commit()

                    event["message_id"] = assistant_msg.id

                # SSE Event Formats:
                # 1. Main JSON payload: data: {"type": ...}\n\n
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"Streaming generation error: {e}", exc_info=True)
            err_event = {
                "type": "error",
                "message": "Generation interrupted. Please try again."
            }
            yield f"data: {json.dumps(err_event)}\n\n"

        finally:
            # If aborted/interrupted before normal 'done' event, persist partial answer
            if not done_payload and accumulated_text.strip():
                try:
                    partial_msg = Message(
                        chat_id=chat_id,
                        role="assistant",
                        content=accumulated_text.strip(),
                        source_type="interrupted",
                        grounding_score=0.0
                    )
                    save_db.add(partial_msg)
                    save_db.commit()
                    logger.info(f"Persisted partially generated response ({len(accumulated_text)} chars) for chat {chat_id}")
                except Exception as save_err:
                    logger.error(f"Failed to persist partial interrupted message: {save_err}")
            save_db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
