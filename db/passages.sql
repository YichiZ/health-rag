CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS passages (
  id         integer PRIMARY KEY,
  passage    text NOT NULL,
  embedding  vector(1536) NOT NULL
);

CREATE INDEX IF NOT EXISTS passages_embedding_hnsw
  ON passages USING hnsw (embedding vector_cosine_ops);
