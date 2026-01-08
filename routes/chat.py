from fastapi import APIRouter, HTTPException
from core.models import (
    ChatRequest,
    ChatResponse,
    GraphRAGContext,
    RAGChunk,
    RAGEntity,
    RAGRelationship,
    StructuredChatResponse,
    StructuredContext,
    StructuredNode
)
from services.neo4j_service import Neo4jService
from core.agents import graphrag_agent, sql_graphrag_agent, label_router_agent

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

@router.post("/chat/structured", response_model=StructuredChatResponse)
async def chat_structured(req: ChatRequest):
    try:
        all_labels = neo4j_service.get_all_labels()

        target_labels = label_router_agent(req.question, all_labels)
        print(f"Router selected: {target_labels}")

        rag_context = neo4j_service.structured_graphrag_search(
            query=req.question,
            target_labels=target_labels, 
            top_k=5
        )

        answer = sql_graphrag_agent(req.question, rag_context)

        response_context = StructuredContext(
            nodes=[
                StructuredNode(
                    label=n["label"],
                    data=n["data"],
                    score=n["score"]
                ) for n in rag_context["nodes"]
            ],
            relationships=rag_context["relationships"]
        )

        return StructuredChatResponse(
            answer=answer,
            context=response_context
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
