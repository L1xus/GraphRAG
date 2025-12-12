import os
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.neo4j import Neo4jTools
from core.models import ExtractedEntities

def entities_extraction_agent():
    return Agent(
        name="PDF Entity Extraction Agent",
        model=OpenAIChat(id="gpt-4o"),
        description="Extract entities and relationships from the text chunk.",
        instructions="""
        Analyze the following text and return ONLY valid JSON that matches the schema:
        {
          "entities": [{"name":"...","type":"PERSON|ORGANIZATION|LOCATION|CONCEPT|DATE|EVENT"}],
          "relationships":[{"from_entity":"...","to_entity":"...","type":"..."}]
        }
        Notes:
         - Use the key names EXACTLY as in the schema.
         - Respond ONLY with JSON (no surrounding commentary).
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
