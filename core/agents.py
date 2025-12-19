import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.neo4j import Neo4jTools
from core.models import ExtractedEntities

def entities_extraction_agent():
    return Agent(
        name="PDF Entity Extraction Agent",
        model=OpenAIChat(id="gpt-4o"),
        description="Extract meaningful entities and relationships from the text chunk.",
        instructions = """
        You are an information extraction assistant specializing in identifying key concepts and relationships.
        
        Task:
        - Analyze ONLY the given text chunk.
        - Extract MEANINGFUL entities that represent core concepts, not peripheral details.
        - Extract relationships between entities that are explicitly supported by the text.
        
        Entity Types:
        - PERSON: Named individuals (authors, researchers, historical figures)
        - ORGANIZATION: Companies, institutions, research groups
        - LOCATION: Geographic places, regions, countries
        - CONCEPT: Key ideas, technologies, methodologies, theories (the main focus)
        - EVENT: Significant events, conferences, milestones
        
        Entity Extraction Rules (CRITICAL):
        1. **Minimum length**: Entity names must be at least 3 characters
        2. **No isolated numbers**: Ignore standalone years, dates, or numeric values unless part of a named entity
        3. **No single letters**: Ignore variables like "p", "q", "x", "y" unless part of a formula name
        4. **No generic terms**: Avoid "the system", "the method", "the approach" - use specific names
        5. **Focus on document core**: Prioritize entities from main content over bibliography/references
        6. **Context required**: Only extract entities that have clear contextual meaning
        7. **Avoid acronyms alone**: If you see "CPU", only extract if it's defined/explained in context
        8. **Skip measurements**: Ignore "10 minutes", "5MB", "100 nodes" unless they're named concepts
        
        Examples of GOOD entities:
        - "Bitcoin", "Blockchain", "Proof-of-Work", "Merkle Tree"
        - "Satoshi Nakamoto", "Alan Turing"
        - "Byzantine Generals Problem", "Double-Spending Attack"
        - "Peer-to-Peer Network", "Cryptographic Hash Function"
        
        Examples of BAD entities (DO NOT EXTRACT):
        - "p", "q", "n", "x" (single variables)
        - "1957", "2008", "April 1980" (standalone dates)
        - "10 minutes", "512 bits" (measurements)
        - "IEEE", "ACM" (unexplained acronyms from citations)
        - Bibliography book titles unless directly discussed in main text
        
        Return ONLY valid JSON:
        {
            "entities": [
                { "name": "...", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|EVENT" }
            ],
            "relationships": [
                { "from_entity": "...", "to_entity": "...", "type": "..." }
            ]
        }
        
        For relationships:
        - "type" MUST be a specific verb that describes the connection, for example:
          - "proposed_by", "created_by", "invented_by"
          - "solves", "addresses", "prevents"
          - "uses", "implements", "relies_on"
          - "is_part_of", "contains", "consists_of"
          - "enables", "requires", "depends_on"
          - "describes", "defines", "explains"
        - Do NOT use generic labels like "related", "connected", "associated"
        - Only create relationships where the connection is clear and explicit
        
        Important:
        - Use only information in the chunk; do NOT rely on outside knowledge
        - If you're unsure about an entity's relevance, skip it
        - Prioritize quality over quantity - 5 meaningful entities beat 50 noisy ones
        """,
        output_schema=ExtractedEntities
    )

def _build_neo4j_tool():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    kwargs = {}
    if uri:
        kwargs["uri"] = uri
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password

    return Neo4jTools(**kwargs)

def query_knowledge_graph(question: str):
    try:
        neo4j_tool = _build_neo4j_tool()
        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[neo4j_tool],
            instructions=[
                "You are a graph database assistant. Use only the Neo4j tool to fetch facts.",
                "Do NOT invent facts. If the graph lacks the info, say so."
            ],
            debug_mode=True
        )
        response = agent.run(question)
        return response.content
    except Exception as e:
        print(f"Graph Agent Error: {e}")
        return None
