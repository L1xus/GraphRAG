import os
from dotenv import load_dotenv
from services.neo4j_service import Neo4jService
from core.sql_load import load_structured_data

load_dotenv()

db_params = {
    'dbname': os.getenv('POSTGRES_DB', 'movies_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

neo4j = Neo4jService()
load_structured_data(db_params, neo4j)
neo4j.close()

