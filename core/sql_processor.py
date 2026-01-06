import psycopg2
from typing import Dict, Any

def get_postgres_schema(db_params: Dict[str, Any]):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = [row[0] for row in cursor.fetchall()]

    schema_summary = ""

    for table in tables:
        schema_summary += f"\n--- TABLE: {table} ---\n"
        
        cursor.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table}'
        """)
        columns = cursor.fetchall()
        schema_summary += "Columns: " + ", ".join([f"{c[0]} ({c[1]})" for c in columns]) + "\n"

        cursor.execute(f'SELECT * FROM "{table}" LIMIT 3')
        rows = cursor.fetchall()
        schema_summary += "Sample Data:\n"
        for row in rows:
            schema_summary += f"  {str(row)}\n"

    conn.close()
    return schema_summary

def fetch_table_data(db_params: Dict, table_name: str):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor(name='fetch_large_result')
    cursor.execute(f'SELECT * FROM "{table_name}"')
    
    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        for row in rows:
            yield row
            
    cursor.close()
    conn.close()
