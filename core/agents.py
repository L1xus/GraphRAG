from agno.agent import Agent
from agno.models.openai import OpenAIChat
from core.models import ExtractedEntities

def entities_extraction_agent():
    return Agent(
        name="PDF Entity Extraction Agent",
        model=OpenAIChat(id="gpt-4o"),
        description="Extract meaningful entities and rich, context-specific relationships from the text chunk.",
        instructions = """
        You are an expert information extraction assistant specializing in building high-quality knowledge graphs.
        Your goal is to extract entities and RICH, CONTEXT-SPECIFIC relationships that capture the actual meaning from the text.

        ENTITY EXTRACTION RULES
        Entity Types:
        - PERSON: Named individuals (e.g., "Satoshi Nakamoto", "Alan Turing")
        - ORGANIZATION: Companies, institutions (e.g., "Bitcoin Network", "IEEE")
        - LOCATION: Geographic places (e.g., "Silicon Valley", "United States")
        - CONCEPT: Key ideas, technologies, methods (e.g., "Proof-of-Work", "Double-Spending")
        - EVENT: Significant events, milestones (e.g., "Bitcoin Genesis Block", "2008 Financial Crisis")
        - TECHNOLOGY: Specific technical systems (e.g., "SHA-256", "Merkle Tree", "Blockchain")
        - PROBLEM: Issues or challenges (e.g., "Byzantine Generals Problem", "Centralization Risk")
        
        Quality Rules:
        ✓ Minimum 3 characters (except well-known acronyms like "CPU", "API")
        ✓ Use specific names, not generic terms ("Bitcoin" not "the system")
        ✓ Include technical terms that are defined/explained in context
        ✓ Extract domain-specific concepts that carry meaning
        
        ✗ NO standalone dates/numbers ("2008", "10 minutes", "1MB")
        ✗ NO single variables ("x", "y", "p", "q")
        ✗ NO generic terms ("the method", "the approach", "the system")
        ✗ NO unexplained acronyms from citations
        
        RELATIONSHIP EXTRACTION RULES (**CRITICAL**)
        
        Your relationships MUST capture the ACTUAL SEMANTIC MEANING from the text.
        Think: "What is the specific action, purpose, or connection described here?"
        
        GUIDELINES FOR RICH RELATIONSHIPS:
        1. **Be Specific and Descriptive**
        2. **Capture the Action or Purpose**
        3. **Include Domain Context**
        4. **Show Directionality Clearly**
        5. **Multi-word Relationship Types**
        6. **Only Extract Explicit Relationships**
        
        OUTPUT FORMAT:
        Return ONLY valid JSON:
        {
            "entities": [
                { "name": "Bitcoin", "type": "TECHNOLOGY" },
                { "name": "Satoshi Nakamoto", "type": "PERSON" },
                { "name": "Double-Spending Attack", "type": "PROBLEM" }
            ],
            "relationships": [
                { 
                    "from_entity": "Satoshi Nakamoto", 
                    "to_entity": "Bitcoin", 
                    "type": "invented" 
                },
                { 
                    "from_entity": "Bitcoin", 
                    "to_entity": "Double-Spending Attack", 
                    "type": "prevents_through_proof_of_work" 
                }
            ]
        }

        Analyze ONLY this chunk and extract the knowledge graph:
        """,
        output_schema=ExtractedEntities
    )

def graphrag_agent(question: str, context_chunks: list, entities: list, relationships: list):
    chunks_text = "\n\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks)])
    entities_text = "\n".join([f"- {e['name']} ({e['type']})" for e in entities])
    relationships_text = "\n".join([f"- {r['from']} --[{r['type']}]--> {r['to']}" for r in relationships])
    
    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=f"""
        You are a GraphRAG assistant. Use the provided context to answer the question.

        CONTEXT FROM VECTOR SEARCH (Text Chunks):
        {chunks_text}

        CONTEXT FROM KNOWLEDGE GRAPH:

        Entities:
        {entities_text}

        Relationships:
        {relationships_text}

        Rules:
        1. Use the text chunks AND the graph structure to answer.
        2. DO NOT repeat or quote the chunk text.
        3. DO NOT print the context or chunk labels in your answer.
        4. If useful, you may reference chunks like (Chunk 1), but do NOT paste their content.
        5. If the answer is not in the context, say "I don't have enough information to answer that."
        6. Do NOT make up information.

        Only return the final answer. No meta commentary.
        Question: {question}
        """,
        markdown=True
    )
    
    try:
        response = agent.run(question)
        return response.content
    except Exception as e:
        print(f"❌ GraphRAG Agent Error: {e}")
        return f"Error generating answer: {str(e)}"
