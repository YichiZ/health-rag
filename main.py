import argparse
import anthropic
import os
from anthropic.types import TextBlock
from dotenv import load_dotenv
import openai
from openai.types import embedding
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from pgvector import Vector
from pathlib import Path

DB_CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/videorag"

def connect():
    conn = psycopg.connect(os.environ.get("DB_CONNECTION_STRING", DB_CONNECTION_STRING))
    register_vector(conn)
    return conn

def main():
    print("Hello from health-rag!")
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--query")
    args = parser.parse_args()
    if args.ingest:
        ingest()
    if args.query is not None:
        query(args.query)

def ingest():
    # id is the parquet index, not a column.
    df = pd.read_parquet("data/rag-mini-bioasq/text-corpus.parquet").reset_index()

    with connect() as conn:
        i = 0
        for row in df.itertuples(index=False):
            passage_id = int(row.id)
            passage = row.passage
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=passage,
            )

            embedding = response.data[0].embedding
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO passages (id, passage, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET passage = EXCLUDED.passage,
                        embedding = EXCLUDED.embedding
                    """,
                    (passage_id, passage, embedding),
                )
            conn.commit()

def ingest_batch():
    pass

def query(text: str):
    print(text)

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    embedding = Vector(response.data[0].embedding)

    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, passage, embedding <=> %s AS distance
                FROM passages
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (embedding, embedding, 5)
            )
            rows = cur.fetchall()
            
            path = Path("output/query-results.txt")
            with path.open("w") as f:
                for passage_id, passage, distance in rows:
                    f.write(f"{passage_id}\t{distance}\n{passage}\n\n")


if __name__ == "__main__":
    main()
