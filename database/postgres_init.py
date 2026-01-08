import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
import numpy as np

def create_tables(cursor):
    create_actors_query = """
    CREATE TABLE IF NOT EXISTS actors (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255),
        date_of_birth DATE,
        place_of_birth TEXT,
        oscars INTEGER,
        oscar_nominations INTEGER,
        bafta INTEGER,
        bafta_nominations INTEGER,
        golden_globes INTEGER,
        golden_globe_nominations INTEGER
    );
    """
    cursor.execute(create_actors_query)
    print("✓ Table 'actors' created!")

    create_movies_query = """
    CREATE TABLE IF NOT EXISTS movies (
        id SERIAL PRIMARY KEY,
        poster_link TEXT,
        series_title VARCHAR(500),
        released_year INTEGER,
        certificate VARCHAR(50),
        runtime VARCHAR(50),
        genre VARCHAR(200),
        imdb_rating DECIMAL(3,1),
        overview TEXT,
        meta_score INTEGER,
        director VARCHAR(200),
        star1 VARCHAR(200),
        star2 VARCHAR(200),
        star3 VARCHAR(200),
        star4 VARCHAR(200),
        no_of_votes INTEGER,
        gross VARCHAR(50)
    );
    """
    cursor.execute(create_movies_query)
    print("✓ Table 'movies' created!")

def process_actors(conn, csv_path):
    cursor = conn.cursor()
    try:
        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=['NULL', ''])

        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        if 'date_of_birth' in df.columns:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')

        int_cols = ['oscars', 'oscar_nominations', 'bafta', 'bafta_nominations', 'golden_globes', 'golden_globe_nominations']
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace({np.nan: None})

        columns = ['name', 'date_of_birth', 'place_of_birth', 'oscars', 'oscar_nominations', 'bafta', 'bafta_nominations', 'golden_globes', 'golden_globe_nominations']
        
        values = []
        for _, row in df.iterrows():
            row_values = []
            for col in columns:
                val = row.get(col)
                if pd.isna(val) or val == '' or val == 'NULL':
                    row_values.append(None)
                elif isinstance(val, pd.Timestamp):
                    row_values.append(val.date())
                else:
                    row_values.append(val)
            values.append(tuple(row_values))

        insert_query = f"""
            INSERT INTO actors ({', '.join(columns)})
            VALUES %s
        """
        
        print(f"Inserting {len(values)} actors...")
        execute_values(cursor, insert_query, values)
        conn.commit()
        print(f"✓ Successfully loaded {len(values)} actors")

    except Exception as e:
        print(f"✗ Error processing actors: {e}")
        conn.rollback()
    finally:
        cursor.close()

def process_movies(conn, csv_path):
    cursor = conn.cursor()
    try:
        print(f"Reading data from {csv_path}...")
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=['NULL', ''])

        df.columns = df.columns.str.lower().str.replace(' ', '_')

        if 'released_year' in df.columns:
            df['released_year'] = pd.to_numeric(df['released_year'], errors='coerce').replace({np.nan: None})
        
        if 'meta_score' in df.columns:
            df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce').replace({np.nan: None})
            
        if 'imdb_rating' in df.columns:
            df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').replace({np.nan: None})

        if 'no_of_votes' in df.columns:
            df['no_of_votes'] = df['no_of_votes'].astype(str).str.replace(',', '')
            df['no_of_votes'] = pd.to_numeric(df['no_of_votes'], errors='coerce').replace({np.nan: None})

        db_cols = ['poster_link', 'series_title', 'released_year', 'certificate', 'runtime', 'genre', 
                   'imdb_rating', 'overview', 'meta_score', 'director', 'star1', 'star2', 'star3', 'star4', 
                   'no_of_votes', 'gross']

        values = []
        for _, row in df.iterrows():
            row_values = []
            for col in db_cols:
                val = row.get(col)
                if pd.isna(val) or val == '' or val == 'NULL':
                    row_values.append(None)
                else:
                    row_values.append(val)
            values.append(tuple(row_values))

        insert_query = f"""
            INSERT INTO movies ({', '.join(db_cols)})
            VALUES %s
        """

        print(f"Inserting {len(values)} movies...")
        execute_values(cursor, insert_query, values)
        conn.commit()
        print(f"✓ Successfully loaded {len(values)} movies")

    except Exception as e:
        print(f"✗ Error processing movies: {e}")
        conn.rollback()
    finally:
        cursor.close()

def init_db():
    db_params = {
        'dbname': os.getenv('POSTGRES_DB', 'movies_db'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        'host': os.getenv('POSTGRES_HOST', 'postgres'),
        'port': os.getenv('POSTGRES_PORT', '5432')
    }

    try:
        print("Connecting to PostgreSQL...")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        create_tables(cursor)
        conn.commit()
        cursor.close()

        actors_file = 'data/actors.csv' 
        if os.path.exists(actors_file):
            process_actors(conn, actors_file)
        else:
            print(f"Couldn't find actors in {actors_file}!")

        movies_file = 'data/movies.csv'
        if os.path.exists(movies_file):
            process_movies(conn, movies_file)
        else:
            print(f"Couldn't find movies in {movies_file}!")

        conn.close()
        print("✓ Database initialization complete.")

    except psycopg2.Error as e:
        print(f"✗ Database error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    init_db()
