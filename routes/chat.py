from fastapi import APIRouter, HTTPException
from core.models import (
    ChatRequest,
    ChatResponse,
    GraphRAGContext,
    RAGChunk,
    RAGEntity,
    RAGRelationship
)
from services.neo4j_service import Neo4jService
from core.agents import graphrag_agent

router = APIRouter()

neo4j_service = Neo4jService()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        question = req.question

        rag_context = neo4j_service.graphrag_search(question, top_k=5)

        answer = graphrag_agent(
            question=question,
            context_chunks=[c["text"] for c in rag_context["chunks"]],
            entities=rag_context["entities"],
            relationships=rag_context["relationships"]
        )

        return ChatResponse(
            answer=answer,
            context=GraphRAGContext(
                chunks=[
                    RAGChunk(
                        id=c.get("id"),
                        text=c["text"],
                        score=c.get("score")
                    )
                    for c in rag_context["chunks"]
                ],
                entities=[
                    RAGEntity(**e)
                    for e in rag_context["entities"]
                ],
                relationships=[
                    RAGRelationship(
                        id=r.get("id"),
                        type=r["type"],
                        source=r["from"],
                        target=r["to"],
                        properties=r.get("properties")
                    )
                    for r in rag_context["relationships"]
                ]
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
