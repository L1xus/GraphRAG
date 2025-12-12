import os
import time
from neo4j import GraphDatabase


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
        """Create a relationship between two entities"""
        with self.driver.session() as session:
            session.run("""
                MATCH (e1:Entity {name: $from_name})
                MATCH (e2:Entity {name: $to_name})
                MERGE (e1)-[r:RELATES_TO {type: $rel_type}]->(e2)
            """, from_name=from_entity, to_name=to_entity, rel_type=rel_type)
    
   
    def search_graph(self, search_text: str, limit: int = 5):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
                WHERE toLower(e.name) CONTAINS toLower($search_text)
                OR toLower(d.content) CONTAINS toLower($search_text)
                OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)
                RETURN DISTINCT 
                    e.name as entity_name,
                    e.type as entity_type,
                    d.filename as doc_filename,
                    d.content as doc_content,
                    collect({
                        related_entity: e2.name,
                        relationship: r.type
                    }) as relationships
                LIMIT $limit
            """, search_text=search_text, limit=limit)

            results = []
            for record in result:
                results.append({
                    'entity': record['entity_name'],
                    'type': record['entity_type'],
                    'document': record['doc_filename'],
                    'content': record['doc_content'][:500],
                    'relationships': [r for r in record['relationships'] if r['related_entity']]
                })

            return results
    
    def get_document_context(self, doc_id: str):
        """Get full context for a document including all entities and relationships"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(d)
                OPTIONAL MATCH (e)-[r:RELATES_TO]->(e2:Entity)
                RETURN 
                    d.filename as filename,
                    d.content as content,
                    collect(DISTINCT {
                        entity: e.name,
                        type: e.type
                    }) as entities,
                    collect(DISTINCT {
                        from: e.name,
                        to: e2.name,
                        type: r.type
                    }) as relationships
            """, doc_id=doc_id)
            
            record = result.single()
            if not record:
                return None
            
            return {
                'filename': record['filename'],
                'content': record['content'],
                'entities': [e for e in record['entities'] if e['entity']],
                'relationships': [r for r in record['relationships'] if r['from'] and r['to']]
            }
