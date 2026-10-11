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
TEXT_CORPUS_PATH = "data/enelpol-rag-mini-bioasq/text-corpus/train-00000-of-00001.parquet"

def connect():
    conn = psycopg.connect(os.environ.get("DB_CONNECTION_STRING", DB_CONNECTION_STRING))
    register_vector(conn)
    return conn

def main():
    print("Hello from health-rag!")
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--ingest-batch", action="store_true")
    parser.add_argument("--ingest-batch-parallel", action="store_true")
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--end-pos", type=int)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--query")
    args = parser.parse_args()
    if args.ingest:
        asyncio.run(ingest_parallel(args.start_pos, args.end_pos))
    if args.ingest_batch:
        ingest_batch(args.start_pos, args.end_pos, args.batch_size)
    if args.ingest_batch_parallel:
        asyncio.run(ingest_batch_parallel(
            args.start_pos, args.end_pos, args.batch_size, args.window_size
        ))
    if args.query is not None:
        query(args.query)

def ingest():
    # id is the parquet index, not a column.
    df = pd.read_parquet(TEXT_CORPUS_PATH).reset_index()

    with connect() as conn:
        for row in df.itertuples():
            position = int(row.Index)
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
                    INSERT INTO passages (id, position, passage, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET position = EXCLUDED.position,
                        passage = EXCLUDED.passage,
                        embedding = EXCLUDED.embedding
                    """,
                    (passage_id, position, passage, embedding),
                )
            conn.commit()

import asyncio
import time
from openai import AsyncOpenAI

async def embed_one(client, semaphore: asyncio.Semaphore, position: int, passage_id: int, passage: str):
    async with semaphore:
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=passage,
            )
        except openai.RateLimitError:
            print(f"rate limit at position {position}, passage {passage_id}")
            raise
    return position, passage_id, passage, response.data[0].embedding

def save_passage(conn, log, started: float, position: int, passage_id: int, passage: str, embedding):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO passages (id, position, passage, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET position = EXCLUDED.position,
                passage = EXCLUDED.passage,
                embedding = EXCLUDED.embedding
            """,
            (passage_id, position, passage, embedding),
        )
    elapsed = time.perf_counter() - started
    log.write(f"{position}\t{passage_id}\t{elapsed:.6f}\n")
    log.flush()
    conn.commit()

async def ingest_parallel(start_pos: int = 0, end_pos: int | None = None, window_size: int = 4):
    started = time.perf_counter()
    df = (
        pd.read_parquet(TEXT_CORPUS_PATH)
        .reset_index()
        .iloc[start_pos:end_pos]
    )
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(window_size)

    tasks = [asyncio.create_task(embed_one(client, semaphore, row.Index, int(row.id), row.passage)) 
    for row in df.itertuples()]

    path = Path("output/ingest-times.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn, path.open("w") as log:
        for finished in asyncio.as_completed(tasks):
            position, passage_id, passage, embedding = await finished
            save_passage(conn, log, started, position, passage_id, passage, embedding)

def ingest_batch(start_pos: int = 0, end_pos: int | None = None, batch_size: int = 100):
    started = time.perf_counter()
    df = (
        pd.read_parquet(TEXT_CORPUS_PATH)
        .reset_index()
        .iloc[start_pos:end_pos]
    )
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    rows = list(df.itertuples())

    path = Path("output/ingest-times.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn, path.open("w") as log:
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            batch_start = int(batch[0].Index)
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[row.passage for row in batch],
                )
            except openai.APIError:
                print(f"batch failed at position {batch_start}")
                raise

            embeddings = {item.index: item.embedding for item in response.data}
            for i, row in enumerate(batch):
                save_passage(
                    conn,
                    log,
                    started,
                    int(row.Index),
                    int(row.id),
                    row.passage,
                    embeddings[i],
                )

async def embed_batch(client, semaphore: asyncio.Semaphore, batch):
    async with semaphore:
        try:
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=[row.passage for row in batch],
            )
        except openai.APIError:
            batch_start = int(batch[0].Index)
            print(f"batch failed at position {batch_start}")
            raise
    return batch, response

async def ingest_batch_parallel(
    start_pos: int = 0,
    end_pos: int | None = None,
    batch_size: int = 100,
    window_size: int = 4,
):
    started = time.perf_counter()
    df = (
        pd.read_parquet(TEXT_CORPUS_PATH)
        .reset_index()
        .iloc[start_pos:end_pos]
    )
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    rows = list(df.itertuples())
    batches = [rows[offset : offset + batch_size] for offset in range(0, len(rows), batch_size)]
    semaphore = asyncio.Semaphore(window_size)
    tasks = [
        asyncio.create_task(embed_batch(client, semaphore, batch))
        for batch in batches
    ]

    path = Path("output/ingest-times.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn, path.open("w") as log:
        for task in tasks:
            batch, response = await task
            embeddings = {item.index: item.embedding for item in response.data}
            for i, row in enumerate(batch):
                save_passage(
                    conn,
                    log,
                    started,
                    int(row.Index),
                    int(row.id),
                    row.passage,
                    embeddings[i],
                )

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
                SELECT position, id, passage, embedding <=> %s AS distance
                FROM passages
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (embedding, embedding, 5)
            )
            rows = cur.fetchall()
            
            path = Path("output/query-results.txt")
            with path.open("w") as f:
                for position, passage_id, passage, distance in rows:
                    f.write(f"{position}\t{passage_id}\t{distance}\n{passage}\n\n")


if __name__ == "__main__":
    main()
