import asyncio
import os
import time
from pathlib import Path

import openai
import pandas as pd
import psycopg
from openai import AsyncOpenAI
from pgvector.psycopg import register_vector

TEXT_CORPUS_PATH = "data/enelpol-rag-mini-bioasq/text-corpus/train-00000-of-00001.parquet"

def connect():
    conn = psycopg.connect(os.environ["DB_CONNECTION_STRING"])
    register_vector(conn)
    return conn

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
    texts = []
    for row in batch:
        passage = row.passage
        if len(passage) > 20_000:
            print(f"truncating passage at position {int(row.Index)} from {len(passage)} chars")
            passage = passage[:20_000]
        texts.append(passage)

    async with semaphore:
        while True:
            try:
                response = await client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts,
                )
                break
            except openai.RateLimitError:
                print(f"rate limit at position {int(batch[0].Index)}, retrying in 60s")
                await asyncio.sleep(60)
            except openai.APIError:
                print(f"batch failed at position {int(batch[0].Index)}")
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
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=0)
    rows = list(df.itertuples())
    batches = [rows[offset : offset + batch_size] for offset in range(0, len(rows), batch_size)]
    semaphore = asyncio.Semaphore(window_size)
    tasks = [
        asyncio.create_task(embed_batch(client, semaphore, batch))
        for batch in batches
    ]

    path = Path("output/ingest-times.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if start_pos else "w"
    with connect() as conn, path.open(log_mode) as log:
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
