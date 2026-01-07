from core.pdf_processor import embed_text
from services.neo4j_service import Neo4jService
from core.agents import sql_graphrag_agent

neo4j = Neo4jService()
query = "What movies did Al Pacino act in and what are they about?"

query_embedding = embed_text([query])[0]

context = neo4j.structured_graphrag_search(
    query_embedding, 
    target_labels=["Movie", "Actor"], 
    top_k=5
)

print(context)
answer = sql_graphrag_agent(query, context)
print(answer)
