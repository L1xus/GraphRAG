from typing import Dict, Any, List
from pypdf import PdfReader
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.knowledge.document.base import Document
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from core.agents import entities_extraction_agent
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file_path: str) -> str:
    try:
        pdf_reader = PdfReader(file_path)

        text_content = []
        for page_index, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(page_text.strip())
            except Exception as e:
                print(f"⚠️ Warning: Cannot extract page {page_index + 1}: {e}")
                continue

        return "\n\n".join(text_content) if text_content else "Empty PDF"
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        raise

def chunk_text(text: str) -> List[str]:
    chunker = AgenticChunking(
        model=OpenAIChat(id="gpt-4o"),
        max_chunk_size=5000,
    )

    try:
        document = Document(content=text)
        chunks = chunker.chunk(document)

        return [chunk.content for chunk in chunks]

    except Exception as e:
        print(f"⚠️ Agentic chunking failed, falling back: {e}")

        fallback_size = 1000
        return [text[i:i + fallback_size] for i in range(0, len(text), fallback_size)]

def embed_text(texts: List[str]):
    if not texts:
        return []
    
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings
        
    except Exception as e:
        print(f"⚠️ Batch embedding failed: {e}")
        print("Falling back to individual embeddings...")
        
        embeddings = []
        for text in texts:
            try:
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"⚠️ Failed to embed individual text: {e}")
                embeddings.append([0.0] * 1536)
        
        return embeddings

def extract_entities_from_chunk(chunk: str, chunk_index: int, previous_chunk: str = None, next_chunk: str = None, extraction_agent: Agent = None) -> Dict[str, Any]:
    context_parts = []
    
    if previous_chunk:
        context_parts.append(f"[PREVIOUS CONTEXT]: {previous_chunk[-500:]}")
    
    context_parts.append(f"[CURRENT CHUNK TO ANALYZE]: {chunk}")
    
    if next_chunk:
        context_parts.append(f"[NEXT CONTEXT]: {next_chunk[:500]}")
    
    full_context = "\n\n".join(context_parts)
    
    additional_instructions = """
    
    IMPORTANT CONTEXT INSTRUCTIONS:
    - You are given the current chunk plus surrounding context (previous/next chunks)
    - Extract entities and relationships ONLY from the [CURRENT CHUNK TO ANALYZE]
    - Use the context sections to better understand references, pronouns, and implicit connections
    - If the current chunk refers to something mentioned in context, use the full proper name
    
    Example:
    [PREVIOUS CONTEXT]: "Bitcoin uses Proof-of-Work consensus..."
    [CURRENT CHUNK]: "This mechanism prevents double-spending by requiring computational work."
    [NEXT CONTEXT]: "The network validates each transaction..."
    
    You should extract:
    - "Proof-of-Work" (mentioned in previous, referenced as "This mechanism")
    - "Double-Spending" 
    - Relationship: "Proof-of-Work" → "prevents" → "Double-Spending"
    """
    
    try:
        # Create a modified prompt that includes context
        result = extraction_agent.run(full_context + additional_instructions)
        
        # Add quality filtering
        extracted_data = result.content
        
        # Filter out low-quality entities
        quality_entities = []
        for entity in (extracted_data.entities or []):
            # Skip very short names or generic terms
            if len(entity.name) < 3:
                continue
            
            # Skip common generic words
            generic_terms = {'the', 'this', 'that', 'these', 'those', 'it', 'its', 
                           'system', 'method', 'approach', 'technique', 'process'}
            if entity.name.lower() in generic_terms:
                continue
            
            quality_entities.append(entity)
        
        # Filter relationships to ensure both entities exist
        entity_names = {e.name for e in quality_entities}
        quality_relationships = []
        
        for rel in (extracted_data.relationships or []):
            # Both entities must exist in our filtered entity list
            if rel.from_entity in entity_names and rel.to_entity in entity_names:
                # Check relationship type is not too generic
                generic_rel_types = {'related', 'connected', 'associated', 'linked', 
                                    'has', 'uses', 'involves'}
                if rel.type.lower() not in generic_rel_types:
                    quality_relationships.append(rel)
        
        return {
            "entities": quality_entities,
            "relationships": quality_relationships
        }
        
    except Exception as e:
        print(f"⚠️ Extraction failed on chunk {chunk_index}: {e}")
        return {"entities": [], "relationships": []}

def extract_entities_per_chunk(chunks: List[str]) -> List[Dict[str, Any]]:
    """Extract entities from each chunk with context awareness"""
    extraction_agent = entities_extraction_agent()
    
    chunk_extractions = []
    
    for i, chunk in enumerate(chunks):
        print(f"\n{'─'*50}")
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        print(f"{'─'*50}")
        
        # Get surrounding context
        previous_chunk = chunks[i-1] if i > 0 else None
        next_chunk = chunks[i+1] if i < len(chunks) - 1 else None
        
        # Extract with context
        data = extract_entities_from_chunk(
            chunk=chunk,
            chunk_index=i,
            previous_chunk=previous_chunk,
            next_chunk=next_chunk,
            extraction_agent=extraction_agent
        )
        
        # Store extraction with chunk reference
        chunk_extractions.append({
            'chunk_index': i,
            'chunk_text': chunk,
            'entities': data['entities'],
            'relationships': data['relationships']
        })
        
        print(f"✓ Extracted {len(data['entities'])} entities, {len(data['relationships'])} relationships")
        
        # Show sample entities for verification
        if data['entities'][:3]:
            print(f"  Sample entities: {', '.join([e.name for e in data['entities'][:3]])}")
        
        # Show sample relationships for verification
        if data['relationships'][:2]:
            print("  Sample relationships:")
            for rel in data['relationships'][:2]:
                print(f"    • {rel.from_entity} --[{rel.type}]--> {rel.to_entity}")
    
    return chunk_extractions

def merge_entity_relationships(chunk_extractions: List[Dict]) -> Dict[str, Any]:
    """Merge entities and relationships across all chunks"""
    entities_map = {}
    
    relationships_set = set()
    relationships_list = []
    
    # Process each chunk's extractions
    for extraction in chunk_extractions:
        for entity in extraction['entities']:
            # Normalize entity name for deduplication
            normalized_name = entity.name.strip()
            
            # Use normalized name as key, but keep original casing
            if normalized_name.lower() not in [e.lower() for e in entities_map.keys()]:
                entities_map[normalized_name] = entity
            else:
                pass
    
    # Collect all relationships
    for extraction in chunk_extractions:
        for rel in extraction['relationships']:
            # Normalize relationship tuple for deduplication
            rel_tuple = (
                rel.from_entity.strip(), 
                rel.to_entity.strip(), 
                rel.type.lower().strip()
            )
            
            if rel_tuple not in relationships_set:
                relationships_set.add(rel_tuple)
                relationships_list.append(rel)
    
    unique_entities = list(entities_map.values())
    
    print("📊 EXTRACTION SUMMARY")
    print(f"  Total unique entities: {len(unique_entities)}")
    print(f"  Total unique relationships: {len(relationships_list)}")
    
    # Show entity type distribution
    from collections import Counter
    entity_types = Counter([e.type for e in unique_entities])
    print("\n  Entity Types:")
    for entity_type, count in entity_types.most_common():
        print(f"    • {entity_type}: {count}")
    
    # Show top entities by connection count
    entity_connections = Counter()
    for rel in relationships_list:
        entity_connections[rel.from_entity] += 1
        entity_connections[rel.to_entity] += 1
    
    print("\n  Most Connected Entities:")
    for entity_name, conn_count in entity_connections.most_common(10):
        print(f"    • {entity_name}: {conn_count} connections")
    
    
    return {
        'entities': unique_entities,
        'relationships': relationships_list
    }
