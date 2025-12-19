import os
import time
from neo4j import GraphDatabase
from core.pdf_processor import embed_text
import numpy as np
from typing import List
import re

class Neo4jService:
    def __init__(self):
        self.driver = None
        self._connect()
    
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
            print("Indexes created")
    
    def create_document_node(self, doc_id: str, filename: str, content: str):
        """Create a document node in the graph"""
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.filename = $filename,
                    d.content = $content,
                    d.created_at = datetime()
            """, doc_id=doc_id, filename=filename, content=content)

    def create_chunk_node(self, chunk_id: str, text: str, embedding: List[float], doc_id: str):
        """store embedding as list property"""
        with self.driver.session() as session:
            session.run("""
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.embedding = $embedding
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, chunk_id=chunk_id, text=text, embedding=embedding, doc_id=doc_id)  

    def create_entity(self, entity_name: str, entity_type: str, doc_id: str):
        """Create an entity node and link it to the document"""
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                WITH e
                MATCH (d:Document {id: $doc_id})
                MERGE (e)-[:MENTIONED_IN]->(d)
            """, name=entity_name, type=entity_type, doc_id=doc_id)
    
    def create_relationship(self, from_entity: str, to_entity: str, rel_type: str):
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
    
    def search_graph(self, search_text: str, limit: int = 10):
        with self.driver.session() as session:
            result = session.run("""
                // Find entities matching the search term
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($search_text)
                
                // Get the document they're mentioned in
                MATCH (e)-[:MENTIONED_IN]->(d:Document)
                
                // Count relationships (for ranking)
                OPTIONAL MATCH (e)-[r]-(connected:Entity)
                WITH e, d, count(DISTINCT r) as rel_count, collect(DISTINCT connected) as connected_entities
                
                // Get outgoing relationships
                OPTIONAL MATCH (e)-[r_out]->(e2:Entity)
                WITH e, d, rel_count, connected_entities,
                     collect(DISTINCT {entity: e2.name, type: type(r_out), direction: 'outgoing'}) as out_rels
                
                // Get incoming relationships
                OPTIONAL MATCH (e1:Entity)-[r_in]->(e)
                WITH e, d, rel_count, connected_entities, out_rels,
                     collect(DISTINCT {entity: e1.name, type: type(r_in), direction: 'incoming'}) as in_rels
                
                // Find relevant chunks
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower(e.name)
                WITH e, d, rel_count, out_rels, in_rels,
                     collect(c.text)[0..2] as sample_chunks
                
                // Return results ranked by connection count
                RETURN DISTINCT 
                    e.name as entity_name,
                    e.type as entity_type,
                    d.filename as doc_filename,
                    d.id as doc_id,
                    rel_count as relationship_count,
                    out_rels + in_rels as relationships,
                    sample_chunks as context_chunks
                ORDER BY rel_count DESC, entity_name
                LIMIT $limit
            """, search_text=search_text, limit=limit)

            results = []
            for record in result:
                # Clean up relationships
                relationships = [
                    {
                        'entity': r['entity'],
                        'type': r['type'],
                        'direction': r['direction']
                    }
                    for r in record['relationships'] if r['entity']
                ]
                
                results.append({
                    'entity': record['entity_name'],
                    'type': record['entity_type'],
                    'document': record['doc_filename'],
                    'doc_id': record['doc_id'],
                    'relationship_count': record['relationship_count'],
                    'relationships': relationships,
                    'context': record['context_chunks']
                })

            return results

    def fetch_all_chunks_with_embeddings(self):
        """Return list of chunks with id, text and embedding"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                RETURN c.id AS id, c.text AS text, c.embedding AS embedding, 
                       d.id as doc_id, d.filename as filename
            """)
            rows = []
            for rec in result:
                emb = rec["embedding"]
                rows.append({
                    "id": rec["id"],
                    "text": rec["text"],
                    "embedding": np.array(emb, dtype=float) if emb else None,
                    "doc_id": rec["doc_id"],
                    "filename": rec["filename"]
                })
            return rows

    def semantic_search(self, query: str, top_k: int = 5):
        """Semantic search using vector similarity"""
        query_embedding = embed_text([query])[0]
        
        chunks = self.fetch_all_chunks_with_embeddings()
        
        results = []
        for chunk in chunks:
            if chunk["embedding"] is not None:
                similarity = np.dot(query_embedding, chunk["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk["embedding"])
                )
                results.append({
                    "text": chunk["text"],
                    "similarity": similarity,
                    "doc_id": chunk["doc_id"],
                    "filename": chunk["filename"]
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
