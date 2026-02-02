"""
Context persistence layer - Saves and retrieves research sessions for multi-turn workflows.
Allows users to tweak datasets and generate additional queries based on previous context.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from vector_store import AsyncVectorStore
from uuid import uuid4


class ResearchContextManager:
    """Manages persistent research context across sessions"""
    
    def __init__(self, vector_store: AsyncVectorStore):
        self.vector_store = vector_store
    
    async def save_research_session(self, session_data: Dict[str, Any]) -> str:
        """
        Save a completed research session to vector store.
        Stores queries, results, and extracted items for future reference.
        """
        session_id = session_data.get("session_id")
        
        # Save initial prompt
        if session_data.get("initial_prompt"):
            await self.vector_store.add_document(
                content=session_data["initial_prompt"],
                metadata={"type": "initial_prompt", "timestamp": datetime.now().isoformat()},
                session_id=session_id
            )
        
        # Save all queries executed
        if session_data.get("queries"):
            queries_text = "\n".join(session_data["queries"])
            await self.vector_store.add_document(
                content=queries_text,
                metadata={
                    "type": "queries",
                    "count": len(session_data["queries"]),
                    "timestamp": datetime.now().isoformat()
                },
                session_id=session_id
            )
        
        # Save extracted items for reference
        if session_data.get("extracted_items"):
            # Save a summary of extracted items
            extracted_text = json.dumps(session_data["extracted_items"][:10], indent=2)  # First 10 items
            await self.vector_store.add_document(
                content=extracted_text,
                metadata={
                    "type": "extracted_items",
                    "count": len(session_data["extracted_items"]),
                    "timestamp": datetime.now().isoformat()
                },
                session_id=session_id
            )
        
        # Save final dataset summary
        if session_data.get("final_dataset"):
            dataset_summary = {
                "total_records": len(session_data["final_dataset"]),
                "fields": list(session_data["final_dataset"][0].keys()) if session_data["final_dataset"] else []
            }
            await self.vector_store.add_document(
                content=json.dumps(dataset_summary),
                metadata={
                    "type": "dataset_summary",
                    "timestamp": datetime.now().isoformat()
                },
                session_id=session_id
            )
        
        return session_id
    
    async def load_session_context(self, session_id: str) -> Dict[str, Any]:
        """Load previous session context"""
        return await self.vector_store.get_session_context(session_id)
    
    async def search_similar_sessions(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar research sessions to inform new queries"""
        return await self.vector_store.search_documents(query, limit=limit)
    
    async def get_previous_queries(self, session_id: str) -> List[str]:
        """Get all queries from a previous session"""
        context = await self.load_session_context(session_id)
        
        previous_queries = []
        for doc in context.get("documents", []):
            if doc.get("metadata", {}).get("type") == "queries":
                # Parse queries from the stored text
                try:
                    # Queries are stored as newline-separated text
                    queries = [q.strip() for q in doc.get("content", "").split("\n") if q.strip()]
                    previous_queries.extend(queries)
                except:
                    pass
        
        return previous_queries
    
    async def get_previous_extracted_items(self, session_id: str) -> List[Dict]:
        """Get extracted items from previous session for comparison"""
        context = await self.load_session_context(session_id)
        
        items = []
        for doc in context.get("documents", []):
            if doc.get("metadata", {}).get("type") == "extracted_items":
                try:
                    items = json.loads(doc.get("content", "[]"))
                    break
                except:
                    pass
        
        return items
    
    async def analyze_data_gaps(self, session_id: str, instructions: str) -> Dict[str, Any]:
        """
        Analyze what data is missing based on user instructions.
        Use vector search to find similar patterns in previous sessions.
        """
        # Search for similar instructions from previous sessions
        similar_docs = await self.vector_store.search_documents(
            instructions,
            limit=5,
            session_id=session_id
        )
        
        return {
            "user_instructions": instructions,
            "similar_patterns": similar_docs,
            "analysis_timestamp": datetime.now().isoformat()
        }


class SessionHistory:
    """Tracks user's research history and patterns"""
    
    def __init__(self, vector_store: AsyncVectorStore):
        self.vector_store = vector_store
        self.sessions_list = []
    
    async def record_session(self, session_id: str, metadata: Dict[str, Any]):
        """Record a new session in history"""
        self.sessions_list.append({
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata
        })
        
        # Also store in vector store for persistence
        await self.vector_store.add_document(
            content=json.dumps(metadata),
            metadata={"type": "session_metadata", "session_id": session_id},
            session_id=session_id
        )
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all recorded sessions"""
        return self.sessions_list
    
    def get_latest_session(self) -> Optional[Dict[str, Any]]:
        """Get the most recent session"""
        return self.sessions_list[-1] if self.sessions_list else None
