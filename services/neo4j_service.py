import os
import time
from neo4j import GraphDatabase
from core.pdf_processor import embed_text
from typing import List, Dict
import re

class Neo4jService:
    def __init__(self):
        self.driver = None
        self._connect()
        self._setup_vector_index()
    
    def _connect(self):
        """Connect to Neo4j with retry logic"""
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                with self.driver.session() as session:
                    session.run("RETURN 1")
                print("✅ Connected to Neo4j")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Failed to connect to Neo4j (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to connect to Neo4j after {max_retries} attempts: {e}")
                    raise
    
    def _setup_vector_index(self):
        """Create vector index for semantic search"""
        with self.driver.session() as session:
            try:
                session.run("""
                    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                print("✅ Vector index created/verified")
            except Exception as e:
                print(f"⚠️ Vector index setup: {e}")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("❌ Neo4j connection closed")
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ Graph cleared")

    def _normalize_rel_type(self, rel_type: str) -> str:
        if not rel_type:
            return "RELATED"

        s = rel_type.strip().upper()
        s = re.sub(r'[^A-Z0-9]+', '_', s)
        s = s.strip('_')
        if not s:
            return "RELATED"
        if s[0].isdigit():
            s = "REL_" + s
        return s
    
    def create_indexes(self):
        """Create indexes for better performance"""
        with self.driver.session() as session:
            session.run("CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            print("✅ Indexes created")
    
    def create_document_node(self, doc_id: str, filename: str, content: str):
        """Create a document node in the graph"""
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.filename = $filename,
                    d.content = $content,
                    d.created_at = datetime()
            """, doc_id=doc_id, filename=filename, content=content)

    def create_chunk_node(self, chunk_id: str, text: str, chunk_index: int, embedding: List[float], doc_id: str):
        """Create chunk node with embedding for vector search"""
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.index = $chunk_index,
                    c.embedding = $embedding
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, chunk_id=chunk_id, text=text, chunk_index=chunk_index, 
                 embedding=embedding, doc_id=doc_id)
    
    def create_chunk_relationships(self, chunk_ids: List[str]):
        """Create NEXT relationships between consecutive chunks"""
        with self.driver.session() as session:
            for i in range(len(chunk_ids) - 1):
                session.run("""
                    MATCH (c1:Chunk {id: $chunk_id_1})
                    MATCH (c2:Chunk {id: $chunk_id_2})
                    MERGE (c1)-[:NEXT]->(c2)
                """, chunk_id_1=chunk_ids[i], chunk_id_2=chunk_ids[i + 1])
        
        print(f"✓ Created {len(chunk_ids) - 1} NEXT relationships between chunks")

    def create_entity(self, entity_name: str, entity_type: str, chunk_id: str):
        """Create an entity node and link it to the chunk it was extracted from"""
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                WITH e
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (e)-[:EXTRACTED_FROM]->(c)
            """, name=entity_name, type=entity_type, chunk_id=chunk_id)
    
    def create_relationship(self, from_entity: str, to_entity: str, rel_type: str):
        """Create relationship between entities"""
        rel_label = self._normalize_rel_type(rel_type)

        cypher = f"""
            MATCH (e1:Entity {{name: $from_name}})
            MATCH (e2:Entity {{name: $to_name}})
            MERGE (e1)-[r:{rel_label}]->(e2)
            SET r.llm_type = $rel_type
        """

        with self.driver.session() as session:
            session.run(
                cypher,
                from_name=from_entity,
                to_name=to_entity,
                rel_type=rel_type,
            )
    
    def vector_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Perform vector similarity search using Neo4j's vector index"""
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_embedding)
                YIELD node AS c, score
                MATCH (c)<-[:HAS_CHUNK]-(d:Document)
                RETURN c.id AS chunk_id, c.text AS text, c.index AS chunk_index,
                       d.id AS doc_id, d.filename AS filename, score
                ORDER BY score DESC
            """, query_embedding=query_embedding, top_k=top_k)
            
            return [dict(record) for record in result]
    
    def get_entities_from_chunks(self, chunk_ids: List[str]) -> List[Dict]:
        """Get all entities extracted from specific chunks"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[:EXTRACTED_FROM]->(c:Chunk)
                WHERE c.id IN $chunk_ids
                RETURN DISTINCT e.name AS name, e.type AS type
            """, chunk_ids=chunk_ids)
            
            return [dict(record) for record in result]
    
    def get_entity_relationships(self, entity_names: List[str], max_hops: int = 2) -> List[Dict]:
        """Get relationships between entities, following connections up to max_hops"""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (e:Entity)
                WHERE e.name IN $entity_names
                MATCH path = (e)-[r*1..{max_hops}]-(connected:Entity)
                RETURN DISTINCT
                    startNode(relationships(path)[0]).name AS from_entity,
                    type(relationships(path)[0]) AS rel_type,
                    endNode(relationships(path)[0]).name AS to_entity
                LIMIT 50
            """, entity_names=entity_names)
            
            relationships = []
            for record in result:
                relationships.append({
                    'from': record['from_entity'],
                    'type': record['rel_type'],
                    'to': record['to_entity']
                })
            
            return relationships
    
    def graphrag_search(self, query: str, top_k: int = 5) -> Dict:
        """Complete GraphRAG search that combines Vector search, Entity extraction and Relationships"""
        query_embedding = embed_text([query])[0]
        vector_results = self.vector_search(query_embedding, top_k)
        
        if not vector_results:
            return {
                'chunks': [],
                'entities': [],
                'relationships': []
            }
        
        chunks = [
            {
                'text': r['text'],
                'score': r['score'],
                'filename': r['filename']
             } 
                  for r in vector_results]
        chunk_ids = [r['chunk_id'] for r in vector_results]
        
        entities = self.get_entities_from_chunks(chunk_ids)
        
        entity_names = [e['name'] for e in entities]
        relationships = self.get_entity_relationships(entity_names, max_hops=2) if entity_names else []
        
        return {
            'chunks': chunks,
            'entities': entities,
            'relationships': relationships
        }
