CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS passages;

CREATE TABLE passages (
  id         integer PRIMARY KEY,
  position   integer NOT NULL,
  passage    text NOT NULL,
  embedding  vector(1536) NOT NULL
);

CREATE INDEX passages_embedding_hnsw
  ON passages USING hnsw (embedding vector_cosine_ops);
