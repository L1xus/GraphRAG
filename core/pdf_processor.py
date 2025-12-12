from typing import Dict, Any, List
from pypdf import PdfReader
from agno.knowledge.chunking.agentic import AgenticChunking
from agno.knowledge.document.base import Document
from agno.models.openai import OpenAIChat
from agno.agent import Agent
from core.agents import entities_extraction_agent
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

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

def embed_text(texts: List[str]) -> List[List[float]]:
    embeddings = []
    for t in texts:
        resp = openai.Embedding.create(input=t, model="text-embedding-3-small")
        emb = resp["data"][0]["embedding"]
        embeddings.append(emb)
    return embeddings

def extract_entities_from_chunk(chunk: str, extraction_agent: Agent) -> Dict[str, Any]:
    try:
        result = extraction_agent.run(chunk)
        return result.content
    except Exception as e:
        print(f"⚠️ Extraction failed on chunk: {e}")
        return {"entities": [], "relationships": []}

def extract_entities_and_relationships(text: str) -> Dict[str, Any]:
    extraction_agent = entities_extraction_agent()

    print("Chunking text using Agno agentic chunking...")
    chunks = chunk_text(text)

    all_entities = []
    all_relationships = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")

        data = extract_entities_from_chunk(chunk, extraction_agent)
        print(f"DATA: {data}")
        all_entities.extend(data.entities or [])
        all_relationships.extend(data.relationships or [])

    entities_map = {e.name: e for e in all_entities}
    unique_entities = list(entities_map.values())

    seen = set()
    unique_rels = []
    for r in all_relationships:
        key = (r.from_entity, r.to_entity, r.type)
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)

    return {
        "entities": unique_entities,
        "relationships": unique_rels,
        "chunks": chunks
    }
