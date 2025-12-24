import uuid
from dotenv import load_dotenv
from services.neo4j_service import Neo4jService
from core.pdf_processor import extract_text_from_pdf, chunk_text, embed_text, extract_entities_per_chunk, merge_entity_relationships
from core.agents import graphrag_agent

load_dotenv()


def load_pipeline(pdf_path: str, filename: str, neo4j_service: Neo4jService):
    print(f"Processing PDF: {filename}")
    
    doc_id = str(uuid.uuid4())
    print(f"Generated Document ID: {doc_id}")
    
    # Step 1: Extract text from PDF
    print("STEP 1: Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    print(f"✓ Extracted {len(text):,} characters from PDF")
    
    if not text or text == "Empty PDF":
        return {
            "success": False,
            "error": "Could not extract text from PDF",
            "doc_id": doc_id
        }
    
    # Step 2: Chunk the text
    print("STEP 2: Chunking text...")
    chunks = chunk_text(text)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings for chunks
    print("STEP 3: Generating embeddings for chunks...")
    try:
        chunk_embeddings = embed_text(chunks)
        print(f"✓ Generated {len(chunk_embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return {
            "success": False,
            "error": f"Embedding generation failed: {str(e)}",
            "doc_id": doc_id
        }
    
    # Step 4: Extract entities PER CHUNK
    print("STEP 4: Extracting entities and relationships from each chunk...")
    chunk_extractions = extract_entities_per_chunk(chunks)
    
    # Merge entities across chunks
    merged_data = merge_entity_relationships(chunk_extractions)
    entities = merged_data['entities']
    relationships = merged_data['relationships']
    
    # Step 5: Store in Neo4j
    print("STEP 5: Storing in Neo4j knowledge graph...")
    
    try:
        # Create document node
        neo4j_service.create_document_node(doc_id, filename, text)
        print(f"✓ Created document node: {filename}")

        # Create chunk nodes with embeddings and store chunk IDs
        chunk_ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            neo4j_service.create_chunk_node(chunk_id, chunk, i, embedding, doc_id)
        print(f"✓ Created {len(chunk_ids)} chunk nodes with embeddings")
        
        # Create sequential relationships between chunks
        neo4j_service.create_chunk_relationships(chunk_ids)
        
        # Create entity nodes linked to their source chunks
        entity_count = 0
        for extraction in chunk_extractions:
            chunk_idx = extraction['chunk_index']
            chunk_id = chunk_ids[chunk_idx]
            
            for entity in extraction['entities']:
                neo4j_service.create_entity(entity.name, entity.type, chunk_id)
                entity_count += 1
                
                if entity_count % 20 == 0:
                    print(f"  ✓ Created {entity_count} entities...")
        
        print(f"✓ Created {entity_count} entity nodes (linked to chunks)")
        
        # Create relationships between entities
        for i, rel in enumerate(relationships):
            neo4j_service.create_relationship(rel.from_entity, rel.to_entity, rel.type)
            if (i + 1) % 20 == 0:
                print(f"  ✓ Created {i + 1}/{len(relationships)} relationships...")
        print(f"✓ Created {len(relationships)} relationships")
        
        # Create indexes
        neo4j_service.create_indexes()
        
        print(f"✅ SUCCESS! Processed '{filename}'")
        print("📊 Statistics:")
        print(f"  • Document ID: {doc_id}")
        print(f"  • Text length: {len(text):,} characters")
        print(f"  • Chunks: {len(chunks)}")
        print(f"  • Entities: {len(entities)}")
        print(f"  • Relationships: {len(relationships)}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "stats": {
                "text_length": len(text),
                "chunks_count": len(chunks),
                "entities_count": len(entities),
                "relationships_count": len(relationships)
            }
        }
        
    except Exception as e:
        print(f"\n❌ Error storing in Neo4j: {e}")
        import traceback
        traceback.print_exc()
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
    
    print("GraphRAG Query:")
    question = "What is Bitcoin and how does it work?"
    print(f"Question: {question}\n")

    rag_context = neo4j_service.graphrag_search(question, top_k=5)

    print("Retrieved Context:")
    print(f"- Chunks: {len(rag_context['chunks'])}")
    print(rag_context['chunks'])
    print(f"- Entities: {len(rag_context['entities'])}")
    print(rag_context['entities'])
    print(f"- Relationships: {len(rag_context['relationships'])}")
    print(rag_context['relationships'])
    print()

    answer = graphrag_agent(
        question=question,
        context_chunks=[c['text'] for c in rag_context['chunks']],
        entities=rag_context['entities'],
        relationships=rag_context['relationships']
    )

    print(f"Answer:\n{answer}\n")
    
    neo4j_service.close()
    print("✅ Done!")

if __name__ == "__main__":
    main()
