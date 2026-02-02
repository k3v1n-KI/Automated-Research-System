-- Enable Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- For generating UUIDs

-- ==========================================
-- 1. AGENT MEMORY (The "Long-Term" Brain)
-- ==========================================

-- This table holds every unique item (hospital, pdf, fact) found.
-- It mixes structured SQL with flexible JSONB and AI Vectors.
CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Searchable Vector (1536 dims for OpenAI small, 768 for Nomic/HuggingFace)
    -- CHANGE TO 768 or 384 if using Local LLM embeddings!
    embedding vector(1536), 
    
    -- The raw extracted data (e.g., { "name": "St. Mikes", "phone": "555-0199" })
    data JSONB NOT NULL,
    
    -- Deduplication Hash (e.g., MD5 of name+address) to prevent exact duplicates
    content_hash TEXT UNIQUE,
    
    -- Metadata for provenance (where did this come from?)
    source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for FAST semantic search (HNSW is faster than IVFFlat)
-- 'vector_cosine_ops' is best for text similarity
CREATE INDEX ON knowledge_base USING hnsw (embedding vector_cosine_ops);

-- Index for FAST JSON filtering (e.g., Find all items where city="Toronto")
CREATE INDEX idx_kb_data ON knowledge_base USING GIN (data);

-- ==========================================
-- 2. OPERATIONAL LOGS (The "Project Manager")
-- ==========================================

CREATE TABLE IF NOT EXISTS research_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    goal TEXT NOT NULL,
    status TEXT DEFAULT 'active', -- 'active', 'completed', 'failed'
    
    -- High-level metrics (e.g., {"items_found": 50, "pages_scraped": 12})
    metrics JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE
);

-- Link items to runs (Many-to-Many, in case multiple runs find the same item)
CREATE TABLE IF NOT EXISTS run_items (
    run_id UUID REFERENCES research_runs(run_id),
    item_id UUID REFERENCES knowledge_base(id),
    PRIMARY KEY (run_id, item_id)
);

-- ==========================================
-- 3. LANGGRAPH CHECKPOINTS (Session State)
-- ==========================================
-- NOTE: LangGraph's PostgresSaver typically creates these automatically via .setup()
-- But defining them here ensures they exist with the correct permissions.

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL, -- The actual serialized state (Graph State)
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL, -- Large data blobs
    PRIMARY KEY (thread_id, checkpoint_ns, type, key)
);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);