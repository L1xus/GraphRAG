from pydantic import BaseModel
from typing import List, Optional

class Entity(BaseModel):
    name: str
    type: str

class Relationship(BaseModel):
    from_entity: str
    to_entity: str
    type: str

class ExtractedEntities(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

class RAGChunk(BaseModel):
    text: str
    score: Optional[float] = None

class RAGEntity(BaseModel):
    name: str
    type: Optional[str] = None

class RAGRelationship(BaseModel):
    type: str
    source: Optional[str] = None
    target: Optional[str] = None

class GraphRAGContext(BaseModel):
    chunks: List[RAGChunk]
    entities: List[RAGEntity]
    relationships: List[RAGRelationship]

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    context: GraphRAGContext
