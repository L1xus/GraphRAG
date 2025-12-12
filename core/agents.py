from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.neo4j import Neo4jTools
from core.models import ExtractedEntities
import os

def entities_extraction_agent():
    return Agent(
        name="PDF Entity Extraction Agent",
        model=OpenAIChat(id="gpt-4o"),
        description="Extracts named entities and relationships from text chunks.",
        instructions="""
        Analyze the following text and extract:
        1. Entities (people, organizations, locations, concepts, dates, events)
        2. Relationships between entities

        Respond ONLY with valid JSON using this structure:
        {
            "entities": [
                {"name": "...", "type": "PERSON|ORGANIZATION|LOCATION|CONCEPT|DATE|EVENT"}
            ],
            "relationships": [
                {"from": "...", "to": "...", "type": "relationship description"}
            ]
        }
        """,
        output_schema=ExtractedEntities,
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
    print(f"Question: {question}")

    try:
        neo4j_tool = _build_neo4j_tool()

        agent = Agent(
            model=OpenAIChat(id="gpt-4o"),
            tools=[neo4j_tool],
            instructions=[
                "You are a graph database assistant.",
                "When answering, prioritize ONLY data retrieved from the Neo4j tools.",
                "Do NOT invent facts. If the graph does not contain enough information, say so clearly."
            ],
            debug_mode=True
        )

        print("Running Agno agent with Neo4jTools...")
        agent_response = agent.run(question)

        return agent_response.content

    except Exception as e:
        print(f"Graph Agent Error: {e}")
