import anthropic
import os
from anthropic.types import TextBlock
from dotenv import load_dotenv
import openai
from openai.types import embedding
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector

DB_CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5432/videorag"

def connect():
    load_dotenv()
    conn = psycopg.connect(os.environ.get("DB_CONNECTION_STRING", DB_CONNECTION_STRING))
    register_vector(conn)
    return conn

def main():
    print("Hello from health-rag!")

    # id is the parquet index, not a column.
    df = pd.read_parquet("data/rag-mini-bioasq/text-corpus.parquet").reset_index()

    with connect() as conn:
        i = 0
        for row in df.itertuples(index=False):
            ingest(conn, int(row.id), row.passage)

            i +=1
            if i > 10:
                break

def ingest(conn, passage_id: int, passage: str):
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


def claude_call():
    client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "content": "Hello, world",
                "role": "user",
            }
        ],
        model="claude-sonnet-5-5",
    )
    text_blocks = [block.text for block in message.content if isinstance(block, TextBlock)]

if __name__ == "__main__":
    main()
