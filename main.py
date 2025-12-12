import uuid
from dotenv import load_dotenv
from services.neo4j_service import Neo4jService
from core.pdf_processor import extract_text_from_pdf, extract_entities_and_relationships
from core.agents import query_knowledge_graph

load_dotenv()


def load_pipeline(pdf_path: str, filename: str, neo4j_service: Neo4jService):
    print(f"Processing PDF: {filename}")
    
    doc_id = str(uuid.uuid4())
    print(f"Generated Document ID: {doc_id}")
    
    print("\nStep 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters from PDF")
    
    if not text or text == "Empty PDF document":
        return {
            "success": False,
            "error": "Could not extract text from PDF",
            "doc_id": doc_id
        }
    
    print("\nStep 2: Extracting entities and relationships with LLM...")
    graph_data = extract_entities_and_relationships(text)
    
    entities = graph_data.get('entities', [])
    relationships = graph_data.get('relationships', [])
    
    print(f"Extracted {len(entities)} unique entities and {len(relationships)} unique relationships")
    
    print("\nStep 3: Storing in Neo4j knowledge graph...")
    
    try:
        # Create document node
        neo4j_service.create_document_node(doc_id, filename, text)
        print(f"✓ Created document node: {filename}")
        
        # Create entity nodes
        for i, entity in enumerate(entities):
            neo4j_service.create_entity(entity.name, entity.type, doc_id)
            if (i + 1) % 10 == 0:
                print(f"  ✓ Created {i + 1}/{len(entities)} entities...")
        print(f"  ✓ Created all {len(entities)} entity nodes")
        
        # Create relationships
        for i, rel in enumerate(relationships):
            neo4j_service.create_relationship(rel.from_entity, rel.to, rel.type)
            if (i + 1) % 10 == 0:
                print(f"  ✓ Created {i + 1}/{len(relationships)} relationships...")
        print(f"  ✓ Created all {len(relationships)} relationships")
        
        # Create indexes
        neo4j_service.create_indexes()
        
        print(f"\n✅ Successfully processed and stored '{filename}' in Neo4j!")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "stats": {
                "text_length": len(text),
                "entities_count": len(entities),
                "relationships_count": len(relationships)
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error storing in Neo4j: {e}")
        return {
            "success": False,
            "error": str(e),
            "doc_id": doc_id
        }


def main():
    neo4j_service = Neo4jService()
    
    pdf_file_path = "data/bitcoin.pdf"
    pdf_filename = "bitcoin.pdf"
    
    # print("Starting PDF loading pipeline...")
    # result = load_pipeline(pdf_file_path, pdf_filename, neo4j_service)
    # print(result)

    # print("Querying the Knowledge Graph")
    # answer = query_knowledge_graph("What is bitcoin?")
    # print(f"ANSWER: {answer}")

    print("Querying the Knowledge Graph")
    answer = neo4j_service.search_graph("bitcoin")
    print(f"ANSWER: {answer}")
    
    neo4j_service.close()
    print("✅ Done!")

if __name__ == "__main__":
    main()
