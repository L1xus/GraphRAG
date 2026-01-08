from pydantic import BaseModel
from typing import Dict, Any, List, Optional

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

class UploadResponseStats(BaseModel):
    text_length: int
    chunks_count: int
    entities_count: int
    relationships_count: int

class UploadResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    stats: Optional[UploadResponseStats] = None
    error: Optional[str] = None

class SQLColumnMapping(BaseModel):
    column_name: str
    target_property: str
    is_embedding_candidate: bool = False
    is_primary_key: bool = False

class SQLNodeMapping(BaseModel):
    source_table: str
    target_label: str
    properties: List[SQLColumnMapping]

class SQLRelationshipMapping(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str

class GraphSchemaMapping(BaseModel):
    nodes: List[SQLNodeMapping]
    relationships: List[SQLRelationshipMapping]

class StructuredNode(BaseModel):
    label: str
    data: Dict[str, Any]
    score: Optional[float] = None

class StructuredContext(BaseModel):
    nodes: List[StructuredNode]
    relationships: List[str]

class StructuredChatResponse(BaseModel):
    answer: str
    context: StructuredContext
