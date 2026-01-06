import psycopg2
from decimal import Decimal
from core.agents import sql_schema_agent
from core.sql_processor import get_postgres_schema, fetch_table_data
from services.neo4j_service import Neo4jService
from core.pdf_processor import embed_text

def load_structured_data(db_params: dict, neo4j_service: Neo4jService):
    print("Starting SQL to Graph Pipeline...")

    # Step 1: Getting database schema
    print("Step 1: Getting Database Schema...")
    schema_context = get_postgres_schema(db_params)

    # Step 2: Semantic Modeling
    print("Step 2: Generating Semantic Graph Mapping with LLM...")
    agent = sql_schema_agent()
    mapping_response = agent.run(schema_context)
    mapping = mapping_response.content
    
    print(f"✓ Generated Mapping: {len(mapping.nodes)} Node Types, {len(mapping.relationships)} Relationship Types")
    
    # Step 3: Labeling Nodes
    print("Step 3: Labeling Graph Nodes...")

    for node_map in mapping.nodes:
        print(f"  Processing Table: {node_map.source_table} -> Label: :{node_map.target_label}")

    for node_map in mapping.nodes:
        print(f"  Processing Table: {node_map.source_table} -> Label: :{node_map.target_label}")
        
        pk_prop = "id"
        for prop in node_map.properties:
            if prop.column_name.lower() == 'id': 
                pk_prop = prop.target_property
                break
        
        print(f"    Using Primary Key: {pk_prop}")

        # Get column names
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        cur.execute(f'SELECT * FROM "{node_map.source_table}" LIMIT 0')
        col_names = [desc[0] for desc in cur.description]
        conn.close()
        
        embed_cols = [p.column_name for p in node_map.properties if p.is_embedding_candidate]
        
        count = 0
        batch_data = []
        
        for row in fetch_table_data(db_params, node_map.source_table):
            row_dict = dict(zip(col_names, row))
            props = {}
            text_to_embed = []
            
            for prop_map in node_map.properties:
                val = row_dict.get(prop_map.column_name)
                
                if val is not None:
                    if isinstance(val, Decimal):
                        val = float(val)
                    
                    props[prop_map.target_property] = val
                    
                    if prop_map.is_embedding_candidate and isinstance(val, str):
                        text_to_embed.append(f"{prop_map.target_property}: {val}")
            
            if text_to_embed:
                props['embedding_text'] = ". ".join(text_to_embed)
            
            batch_data.append(props)
            
            if len(batch_data) >= 100:
                _push_batch(neo4j_service, node_map.target_label, batch_data, embed_cols, pk_prop)
                batch_data = []
                
            count += 1
            
        if batch_data:
            _push_batch(neo4j_service, node_map.target_label, batch_data, embed_cols, pk_prop)
            
        print(f"    ✓ Loaded {count} nodes for {node_map.target_label}")

    # Step 4: Linking
    print(" Step 4: Linking Entities...")
    for rel in mapping.relationships:
        print(f"  Creating Edge: (:{rel.source_table}) -[:{rel.relationship_type}]-> (:{rel.target_table})")
        
        try:
            # Find Source and Target configurations
            source_node = next(n for n in mapping.nodes if n.source_table == rel.source_table)
            target_node = next(n for n in mapping.nodes if n.source_table == rel.target_table)
            
            # Find the property names the LLM chose
            source_prop = next(p.target_property for p in source_node.properties if p.column_name == rel.source_column)
            target_prop = next(p.target_property for p in target_node.properties if p.column_name == rel.target_column)

            neo4j_service.create_structured_relationship(
                source_node.target_label, source_prop, 
                target_node.target_label, target_prop, 
                rel.relationship_type
            )
        except StopIteration:
            print(f"⚠️ Warning: Could not map relationship {rel.relationship_type}")
            continue

    print("✅ SQL Pipeline Complete!")

def _push_batch(service, label, data_list, embed_cols, pk_prop):
    """Batch embed and write"""
    if embed_cols:
        texts = [d['embedding_text'] for d in data_list if 'embedding_text' in d]
        if texts:
            embeddings = embed_text(texts)
            embed_idx = 0
            for d in data_list:
                if 'embedding_text' in d:
                    d['embedding'] = embeddings[embed_idx]
                    del d['embedding_text'] 
                    embed_idx += 1
    
    service.create_structured_nodes_batch(label, data_list, pk_prop)
