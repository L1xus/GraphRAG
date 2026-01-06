from agno.agent import Agent
from agno.models.openai import OpenAIChat
from core.models import ExtractedEntities, GraphSchemaMapping

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

def sql_schema_agent():
    return Agent(
        name="SQL Schema to Graph Mapper",
        model=OpenAIChat(id="gpt-4o"),
        description="Map SQL tables to Knowledge Graph Nodes and Relationships.",
        instructions="""
        You are an expert Data Architect. Your goal is to convert a Relational Schema (SQL) into a Semantic Knowledge Graph.
        
        INPUT DATA:
        You will receive a list of Tables, Columns, and Sample Data.

        YOUR TASK:
        1. **Node Mapping**: Map every table to a Node Label (e.g., 'tbl_users' -> 'User'). 
           - Rename columns to clean properties (e.g., 'dob' -> 'date_of_birth').
           - Identify which text columns (like 'overview', 'bio', 'review') should be EMBEDDED for Vector Search (`is_embedding_candidate: true`).
           - IDs, Dates, and Numbers should NOT be embedded.

        2. **Relationship Mapping**: Identify links between tables.
           - Look for Explicit Foreign Keys.
           - Look for **Implicit Soft Links** (e.g., Table 'Movies' has 'Star1', Table 'Actors' has 'Name'. Map this!).
           - Give relationships active verbs (e.g., 'ACTED_IN', 'DIRECTED', 'PURCHASED').

        OUTPUT:
        Return strict JSON adhering to the `GraphSchemaMapping` schema.
        """,
        output_schema=GraphSchemaMapping
    )

def sql_graphrag_agent(question: str, context: dict):
    nodes_text = ""
    for node in context['nodes']:
        label = node['label']
        data = node['data']
        name = data.get('title') or data.get('name') or data.get('id')
        nodes_text += f"- [{label}] {name}: {data}\n"

    relationships_text = "\n".join(context['relationships'])

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=f"""
        You are a SQL Data Assistant powered by a Knowledge Graph.
        
        CONTEXT FROM DATABASE:
        
        Found Entities (via Vector Search):
        {nodes_text}
        
        Connected Relationships (via Graph Traversal):
        {relationships_text}
        
        INSTRUCTIONS:
        1. Answer the user's question based ONLY on the provided database context.
        2. If you cite a movie or person, mention specific details (e.g., "The Godfather (1972)").
        3. Use the relationship data to explain connections (e.g., "Directed by...").
        4. If the answer is not in the data, say "I cannot find that information in the database."
        
        Question: {question}
        """,
        markdown=True
    )
    
    try:
        response = agent.run(question)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"
