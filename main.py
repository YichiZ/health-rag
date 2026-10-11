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
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--end-pos", type=int)
    parser.add_argument("--query")
    args = parser.parse_args()
    if args.ingest:
        asyncio.run(ingest_parallel(args.start_pos, args.end_pos))
    if args.query is not None:
        query(args.query)

def ingest():
    # id is the parquet index, not a column.
    df = pd.read_parquet("data/rag-mini-bioasq/text-corpus.parquet").reset_index()

    with connect() as conn:
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

import asyncio
import time
from openai import AsyncOpenAI

async def embed_one(client, position: int, passage_id: int, passage: str, parallel_limit: int = 4):
    async with asyncio.Semaphore(parallel_limit):
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=passage,
            )
        except openai.RateLimitError:
            print(f"rate limit at position {position}, passage {passage_id}")
            raise
    return passage_id, passage, response.data[0].embedding

async def ingest_parallel(start_pos: int = 0, end_pos: int | None = None):
    started = time.perf_counter()
    df = (
        pd.read_parquet("data/rag-mini-bioasq/text-corpus.parquet")
        .reset_index()
        .iloc[start_pos:end_pos]
    )
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    tasks = [
        embed_one(client, row.Index, int(row.id), row.passage) for row in df.itertuples()
    ]

    path = Path("output/ingest-times.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn, path.open("w") as log:
        for finished in asyncio.as_completed(tasks):
            passage_id, passage, embedding = await finished
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
            # Perf
            elapsed = time.perf_counter() - started
            log.write(f"{passage_id}\t{elapsed:.6f}\n")
            log.flush()

            # Commit
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
