"""
Vector store for long-term knowledge storage.
Uses Postgres with pgvector extension for embeddings.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

import psycopg
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# Load environment
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

db_url = os.getenv("DATABASE_URL", "postgresql://kevin:kevin@localhost:5432/agent_memory")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class VectorStore:
    """Manage embeddings in Postgres with pgvector"""
    
    def __init__(self, db_url: str = db_url):
        self.db_url = db_url
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        self.conn = psycopg.connect(self.db_url)
        print("✓ Connected to vector store")
        
        # Enable pgvector extension
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
        
        self._create_tables()
    
    def disconnect(self):
        """Disconnect from database"""
        if self.conn:
            self.conn.close()
            print("✓ Disconnected from vector store")
    
    def _create_tables(self):
        """Create necessary tables for vector storage"""
        with self.conn.cursor() as cur:
            # Thoughts table only (agent reasoning steps)
            # Knowledge base already exists in init.sql
            cur.execute("""
                CREATE TABLE IF NOT EXISTS thoughts (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    session_id UUID,
                    step_number INTEGER,
                    type VARCHAR(50),
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Index for thought search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS thoughts_embedding_idx 
                ON thoughts USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            self.conn.commit()
            print("✓ Vector store tables created")
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None, 
                    session_id: Optional[str] = None) -> str:
        """Add document with embedding to knowledge_base"""
        import hashlib
        
        # Generate embedding
        embedding = self._embed(content)
        
        # Generate content hash for deduplication
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        # Insert document into knowledge_base
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO knowledge_base (embedding, data, content_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (content_hash) DO UPDATE SET last_updated_at = CURRENT_TIMESTAMP
                RETURNING id;
            """, (
                embedding.tolist(),
                json.dumps({
                    "content": content,
                    "metadata": metadata or {},
                    "session_id": str(session_id) if session_id else None,
                    "type": "document"
                }),
                content_hash
            ))
            doc_id = cur.fetchone()[0]
            self.conn.commit()
        
        print(f"✓ Added document {doc_id} to knowledge_base")
        return str(doc_id)
    
    def add_thought(self, content: str, session_id: str, step_number: int, 
                   thought_type: str = "reasoning"):
        """Add agent thought/reasoning step to vector store"""
        
        # Generate embedding
        embedding = self._embed(content)
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO thoughts (content, session_id, step_number, type, embedding)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (content, session_id, step_number, thought_type, embedding.tolist()))
            thought_id = cur.fetchone()[0]
            self.conn.commit()
        
        return thought_id
    
    def search_documents(self, query: str, limit: int = 5, session_id: Optional[str] = None) -> List[Dict]:
        """Search knowledge_base by semantic similarity"""
        
        # Embed query
        query_embedding = self._embed(query)
        
        with self.conn.cursor() as cur:
            where_clause = ""
            params = [query_embedding.tolist(), limit]
            
            if session_id:
                where_clause = " WHERE data->>'session_id' = %s"
                params.insert(1, session_id)
            
            cur.execute(f"""
                SELECT id, data, 1 - (embedding <=> %s::vector) as similarity
                FROM knowledge_base
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, params)
            
            results = []
            for row in cur.fetchall():
                doc_data = row[1]
                results.append({
                    "id": str(row[0]),
                    "content": doc_data.get("content", ""),
                    "metadata": doc_data.get("metadata", {}),
                    "similarity": float(row[2])
                })
        
        return results
    
    def search_thoughts(self, query: str, limit: int = 3, session_id: Optional[str] = None) -> List[Dict]:
        """Search agent thoughts for similar reasoning patterns"""
        
        query_embedding = self._embed(query)
        
        with self.conn.cursor() as cur:
            where_clause = ""
            params = [query_embedding.tolist(), limit]
            
            if session_id:
                where_clause = " WHERE session_id = %s"
                params.insert(1, session_id)
            
            cur.execute(f"""
                SELECT id, content, session_id, step_number, type, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM thoughts
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, params)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "session_id": str(row[2]),
                    "step_number": row[3],
                    "type": row[4],
                    "similarity": float(row[5])
                })
        
        return results
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get all documents and thoughts for a session"""
        
        with self.conn.cursor() as cur:
            # Get documents from knowledge_base
            cur.execute("""
                SELECT id, data
                FROM knowledge_base
                WHERE data->>'session_id' = %s
                ORDER BY created_at DESC;
            """, (session_id,))
            
            documents = [
                {"id": str(row[0]), "content": row[1].get("content", ""), "metadata": row[1].get("metadata", {})}
                for row in cur.fetchall()
            ]
            
            # Get thoughts
            cur.execute("""
                SELECT id, content, step_number, type
                FROM thoughts
                WHERE session_id = %s
                ORDER BY step_number;
            """, (session_id,))
            
            thoughts = [
                {"id": row[0], "content": row[1], "step": row[2], "type": row[3]}
                for row in cur.fetchall()
            ]
        
        return {
            "session_id": session_id,
            "documents": documents,
            "thoughts": thoughts,
            "timestamp": datetime.now().isoformat()
        }
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI"""
        
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        
        return np.array(response.data[0].embedding, dtype=np.float32)


# ============================================================================
# Async wrapper for use in Flask/SocketIO
# ============================================================================

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=3)


class AsyncVectorStore:
    """Async wrapper for VectorStore"""
    
    def __init__(self, db_url: str = db_url):
        self.vector_store = VectorStore(db_url)
        self.vector_store.connect()
    
    async def add_document(self, content: str, metadata: Optional[Dict] = None, 
                          session_id: Optional[str] = None) -> int:
        """Async wrapper for add_document"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self.vector_store.add_document,
            content, metadata, session_id
        )
    
    async def add_thought(self, content: str, session_id: str, step_number: int,
                         thought_type: str = "reasoning"):
        """Async wrapper for add_thought"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self.vector_store.add_thought,
            content, session_id, step_number, thought_type
        )
    
    async def search_documents(self, query: str, limit: int = 5, 
                              session_id: Optional[str] = None) -> List[Dict]:
        """Async wrapper for search_documents"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self.vector_store.search_documents,
            query, limit, session_id
        )
    
    async def search_thoughts(self, query: str, limit: int = 3,
                             session_id: Optional[str] = None) -> List[Dict]:
        """Async wrapper for search_thoughts"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self.vector_store.search_thoughts,
            query, limit, session_id
        )
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Async wrapper for get_session_context"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            self.vector_store.get_session_context,
            session_id
        )
    
    def disconnect(self):
        """Close database connection"""
        self.vector_store.disconnect()


if __name__ == "__main__":
    # Test vector store
    store = VectorStore()
    store.connect()
    
    # Add test documents
    store.add_document("Python is a programming language", {"type": "fact"})
    store.add_document("WebSockets enable real-time communication", {"type": "fact"})
    
    # Search
    results = store.search_documents("programming languages")
    for result in results:
        print(f"Found: {result['content']} (similarity: {result['similarity']:.2f})")
    
    store.disconnect()
