from pydantic import BaseModel
from typing import List

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
