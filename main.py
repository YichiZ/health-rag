import argparse
import asyncio
import os
from pathlib import Path

import anthropic
import openai
from anthropic.types import TextBlock
from dotenv import load_dotenv
from pgvector import Vector

from ingest import connect, ingest_batch, ingest_batch_parallel, ingest_parallel

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
                (embedding, embedding, 10)
            )
            rows = cur.fetchall()

            path = Path("output/query-results.txt")
            with path.open("w") as f:
                for position, passage_id, passage, distance in rows:
                    f.write(f"{position}\t{passage_id}\t{distance}\n{passage}\n\n")

            client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))

            context = "\n\n".join(
                f"[{passage_id}] {passage}" for _, passage_id, passage, _ in rows
            )

            system = "Answer the question in one or two sentences using only the passages. If they do not contain the answer, say so."
            user_message = f"Question: {text}\n\nPassages:\n{context}"

            result = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens = 1024,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )

            answer = "\n".join(block.text for block in result.content if isinstance(block, TextBlock))

            print(answer)
            Path("output/answer.txt").write_text(answer + "\n")

if __name__ == "__main__":
    main()
