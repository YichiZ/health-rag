# health-rag

A simple RAG over a passage corpus. Ingest is an exercise in parallelizing and batching calls to the embedding API.

Ingest embeds each passage with OpenAI and stores it in Postgres. A query embeds the question, retrieves the nearest passages, and asks Claude to answer from those passages only.

We train the model on the [RAG mini BioASQ](https://huggingface.co/datasets/enelpol/rag-mini-bioasq) training data, a cleaned subset of [BioASQ](https://www.bioasq.org/) (via [rag-datasets/rag-mini-bioasq](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)). Ingest reads the train passage corpus at `data/enelpol-rag-mini-bioasq/text-corpus/train-00000-of-00001.parquet`.

## How a query runs

1. The question is embedded with `text-embedding-3-small`.
2. Postgres returns the 10 nearest passages by cosine distance (`pgvector`).
3. Those passages are written to `output/query-results.txt`.
4. Claude (`claude-sonnet-5-5`) answers in one or two sentences, using only the retrieved passages.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.14 (uv installs the version in `.python-version`)
- Docker, or a local Postgres with the [pgvector](https://github.com/pgvector/pgvector) extension
- An OpenAI API key (embeddings)
- An Anthropic API key (answers)

## Install

```bash
git clone https://github.com/YichiZ/health-rag.git
cd health-rag
uv sync
```

## Set up the database

Start Postgres with pgvector. The example `DB_CONNECTION_STRING` in `.env.example` is `postgresql://postgres:postgres@localhost:5432/videorag`.

```bash
docker run -d \
  --name health-rag-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=videorag \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

Create the `passages` table and its HNSW index. This script drops `passages` if it already exists.

```bash
docker exec -i health-rag-db psql -U postgres -d videorag < db/passages.sql
```

## Configure

Copy `.env.example` to `.env` in the project root and fill in the keys. `.env` is gitignored.

```bash
cp .env.example .env
```

The app reads `DB_CONNECTION_STRING` from `.env`.

## Index the corpus

Embed the train corpus and store it. This reads `data/enelpol-rag-mini-bioasq/text-corpus/train-00000-of-00001.parquet` (40,221 passages). Three commands exercise the embedding calls:

- `--ingest` sends one passage per request, up to four at a time.
- `--ingest-batch` sends batches of 100, one batch after another.
- `--ingest-batch-parallel` sends batches of 100, four batches at a time. If a batch hits a rate limit, that batch waits 60 seconds and retries until it succeeds.

```bash
uv run python main.py --ingest-batch-parallel
```

Rows already stored are updated, so stopping and starting again is safe. Resume from a parquet row offset, or limit the range:

```bash
uv run python main.py --ingest-batch-parallel --start-pos 20000
uv run python main.py --ingest-batch-parallel --start-pos 0 --end-pos 1000
```

`--batch-size` defaults to 100 and `--window-size` defaults to 4. Passages longer than 20,000 characters are truncated before embedding. Each finished row is logged to `output/ingest-times.txt` as `position`, passage `id`, and elapsed seconds. A run that starts at row 0 overwrites that file. A run with `--start-pos` greater than 0 appends to it.

## Ask a question

```bash
uv run python main.py --query "What is the implication of histone lysine methylation in medulloblastoma?"
```

The answer is printed in the terminal and written to `output/answer.txt`. The retrieved passages are written to `output/query-results.txt`, one block per hit:

```text
position	passage_id	cosine_distance
passage text
```

`output/` is gitignored. Question-answer pairs shipped with the corpus are a useful source of questions. They live in `data/enelpol-rag-mini-bioasq/question-answer-passages/` (4,012 train, 707 test), with `question`, `answer`, `id`, and `relevant_passage_ids`. The query command does not score against those labels.

## Data

| Path | What it is |
| --- | --- |
| `data/enelpol-rag-mini-bioasq/text-corpus/train-00000-of-00001.parquet` | Passages the ingest command indexes (`passage`, `id`) |
| `data/enelpol-rag-mini-bioasq/question-answer-passages/` | BioASQ questions, reference answers, and relevant passage ids |
| `data/rag-mini-bioasq/` | Earlier source corpus, kept for reference |

The dataset is licensed [CC BY 2.5](https://creativecommons.org/licenses/by/2.5/). Cite BioASQ and the Hugging Face dataset pages above when you reuse the passages.
