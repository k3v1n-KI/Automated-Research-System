"""
Flask WebSocket server - Research Algorithm Interface
Streams algorithm execution with real-time progress updates.
Persists context between sessions using Postgres vector store.
Tracks diversity metrics across research iterations.
"""

import asyncio
import json
import os
import csv
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session as flask_session
from flask_socketio import SocketIO, emit, disconnect
from dotenv import find_dotenv, load_dotenv

from algorithm import build_research_algorithm
from vector_store import AsyncVectorStore
from context_persistence import ResearchContextManager, SessionHistory
from diversity_analyzer import analyze_dataset_diversity, generate_diversity_report

# Load environment
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

# Flask app
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "research-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Create datasets folder
DATASETS_DIR = Path(__file__).parent / 'datasets'
DATASETS_DIR.mkdir(exist_ok=True)

# Initialize vector store and context managers
try:
    vector_store = AsyncVectorStore()
    context_manager = ResearchContextManager(vector_store)
    session_history = SessionHistory(vector_store)
    print("Vector store initialized for context persistence")
except Exception as e:
    print(f"Warning: Vector store failed to initialize: {e}")
    vector_store = None
    context_manager = None
    session_history = None

# Session management
sessions = {}  # session_id -> {"algorithm_graph": ..., "state": ..., "client_id": ...}


@app.route("/")
def index():
    """Serve setup page"""
    return render_template('index.html')


@app.route("/chat")
def chat():
    """Serve chat interface with setup data"""
    return render_template('chat_interface.html')


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions)
    }



@socketio.on("connect")
def handle_connect():
    """Client connected"""
    print(f"Client connected: {request.sid}")
    emit("connected", {"timestamp": datetime.now().isoformat()})


@socketio.on("disconnect")
def handle_disconnect():
    """Client disconnected"""
    print(f"Client disconnected: {request.sid}")


@socketio.on("start_research")
def handle_start_research(data):
    """Start research algorithm with optional previous session context"""
    
    client_id = request.sid
    prompt = data.get("prompt", "").strip()
    previous_session_id = data.get("previous_session_id")
    tweak_instructions = data.get("tweak_instructions", "").strip()
    columns = data.get("columns", [])  # List of {name: str, isPriority: bool}
    priority_columns = data.get("priority_columns", [])  # List of column names
    
    if not prompt:
        emit("error", {"content": "Empty prompt"})
        return
    
    print(f"\n{'='*70}")
    print(f"Starting Research Session")
    print(f"   Client: {client_id}")
    print(f"   Prompt: {prompt[:100]}...")
    print(f"   Columns: {[c['name'] for c in columns]}")
    print(f"   Priority Columns: {priority_columns}")
    if previous_session_id:
        print(f"   Previous Session: {previous_session_id}")
        print(f"   Tweaks: {tweak_instructions[:100] if tweak_instructions else 'None'}...")
    print(f"{'='*70}")
    
    # Create session
    session_id = str(uuid4())
    
    def emit_progress(event_type, data):
        """Emit progress to client"""
        socketio.emit(event_type, data, room=client_id)
    
    # Build algorithm with progress emitter
    algorithm_graph = build_research_algorithm(emit_fn=emit_progress)
    
    # Create initial state dict matching ResearchState schema
    state = {
        'initial_prompt': prompt,
        'column_specs': [],
        'queries': [],
        'search_results': [],
        'validated_urls': [],
        'scraped_content': [],
        'extracted_items': [],
        'final_dataset': [],
        'session_id': session_id,
        'round': 0,
        'error': None,
        'previous_session_id': previous_session_id,
        'tweak_instructions': tweak_instructions,
        'previous_queries': [],
        'previous_items': [],
        'columns': columns,
        'priority_columns': priority_columns
    }
    
    # Load previous context if available
    if previous_session_id and context_manager:
        try:
            # Run async loading in background
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            previous_context = loop.run_until_complete(
                context_manager.load_session_context(previous_session_id)
            )
            previous_queries = loop.run_until_complete(
                context_manager.get_previous_queries(previous_session_id)
            )
            previous_items = loop.run_until_complete(
                context_manager.get_previous_extracted_items(previous_session_id)
            )
            
            state['previous_queries'] = previous_queries
            state['previous_items'] = previous_items
            
            print(f"   Loaded previous context: {len(previous_queries)} queries, {len(previous_items)} items")
            emit("context_loaded", {
                "previous_queries_count": len(previous_queries),
                "previous_items_count": len(previous_items)
            })
        except Exception as e:
            print(f"   Error loading previous context: {e}")
    
    print(f"   State initialized with keys: {list(state.keys())}")
    
    # Store session
    sessions[session_id] = {
        "client_id": client_id,
        "state": state,
        "algorithm_graph": algorithm_graph,
        "started_at": datetime.now().isoformat()
    }
    
    # Emit start
    emit("research_start", {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    })
    
    # Run algorithm in background
    socketio.start_background_task(
        _run_algorithm_task,
        session_id,
        client_id,
        state,
        algorithm_graph
    )


def _run_algorithm_task(session_id: str, client_id: str, state: dict, algorithm_graph):
    """Run algorithm asynchronously"""
    try:
        print(f"\n{'='*70}")
        print(f"🔧 Running Algorithm Task")
        print(f"   Session: {session_id}")
        print(f"   State keys: {list(state.keys())}")
        print(f"   Prompt: {state.get('initial_prompt', 'MISSING')[:50]}...")
        print(f"   Columns: {state.get('columns', 'MISSING')}")
        print(f"   Priority Columns: {state.get('priority_columns', 'MISSING')}")
        print(f"{'='*70}\n")
        
        # Run algorithm with state dict
        final_state = asyncio.run(algorithm_graph.ainvoke(state))
        
        # Calculate diversity metrics if priority columns are defined
        dataset = final_state.get('final_dataset', [])
        priority_columns = final_state.get('priority_columns', [])
        columns = final_state.get('columns', [])
        diversity_report = None
        
        # Filter dataset to only include specified columns
        if columns and dataset:
            column_names = [col['name'] for col in columns if isinstance(col, dict)]
            filtered_dataset = []
            for record in dataset:
                if isinstance(record, dict):
                    filtered_record = {k: v for k, v in record.items() if k in column_names}
                    filtered_dataset.append(filtered_record)
            dataset = filtered_dataset
            print(f"Filtered dataset to {len(column_names)} columns: {column_names}")
        
        print(f"\n{'='*70}")
        print(f"✅ Algorithm Complete")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Priority columns from final_state: {priority_columns}")
        print(f"   Type: {type(priority_columns)}")
        print(f"{'='*70}\n")
        
        if priority_columns and dataset:
            try:
                print(f"📊 Generating diversity report...")
                diversity_report = generate_diversity_report(
                    session_id=session_id,
                    records=dataset,
                    priority_columns=priority_columns,
                    round=final_state.get('round', 0)
                )
                print(f"✅ Diversity Report Generated:")
                print(f"  Columns: {priority_columns}")
                print(f"  Diversity Index: {diversity_report.get('overall_diversity_index', 0):.3f}")
                print(f"  Column Analysis: {len(diversity_report.get('column_analysis', []))} columns")
            except Exception as e:
                print(f"❌ Warning: Could not generate diversity report: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Skipping diversity report: priority_columns={bool(priority_columns)}, dataset_size={len(dataset)}")
        
        # Emit results
        socketio.emit("research_complete", {
            "session_id": session_id,
            "final_dataset": dataset,
            "columns": final_state.get('columns', []),
            "priority_columns": priority_columns,
            "diversity_report": diversity_report,
            "statistics": {
                "queries": len(final_state.get('queries', [])),
                "urls_found": len(final_state.get('search_results', [])),
                "urls_validated": len(final_state.get('validated_urls', [])),
                "urls_scraped": len(final_state.get('scraped_content', [])),
                "records_extracted": len(final_state.get('extracted_items', [])),
                "final_count": len(dataset)
            },
            "timestamp": datetime.now().isoformat()
        }, room=client_id)
        
        # Save dataset as CSV
        dataset = final_state.get('final_dataset', [])
        if dataset:
            csv_filename = f"dataset_{session_id[:8]}.csv"
            csv_path = DATASETS_DIR / csv_filename
            
            try:
                # Collect all possible field names from all records
                all_fields = set()
                for record in dataset:
                    if isinstance(record, dict):
                        all_fields.update(record.keys())
                
                all_fields = sorted(list(all_fields))
                
                # Ensure all records have all fields (fill missing with empty string)
                normalized_dataset = []
                for record in dataset:
                    normalized_record = {}
                    for field in all_fields:
                        normalized_record[field] = record.get(field, '') if isinstance(record, dict) else ''
                    normalized_dataset.append(normalized_record)
                
                # Write CSV with all fields
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    if normalized_dataset:
                        writer = csv.DictWriter(f, fieldnames=all_fields)
                        writer.writeheader()
                        writer.writerows(normalized_dataset)
                    
                print(f"Dataset saved to: {csv_path}")
                print(f"  Fields: {all_fields}")
                print(f"  Records: {len(normalized_dataset)}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
        
        # Save session context to vector store for persistence
        if context_manager:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                loop.run_until_complete(
                    context_manager.save_research_session({
                        "session_id": session_id,
                        "initial_prompt": final_state.get('initial_prompt'),
                        "queries": final_state.get('queries', []),
                        "extracted_items": final_state.get('extracted_items', []),
                        "final_dataset": final_state.get('final_dataset', [])
                    })
                )
                
                print(f"Session context saved to vector store")
            except Exception as e:
                print(f"Error saving to vector store: {e}")
        
        print(f"\nResearch completed for session {session_id}")
        print(f"  Final dataset: {len(dataset)} records")
        
    except Exception as e:
        error_msg = str(e)
        print(f"\nResearch failed: {error_msg}")
        socketio.emit("research_error", {
            "session_id": session_id,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }, room=client_id)


@socketio.on("export_dataset")
def handle_export_dataset(data):
    """Export dataset to JSON/CSV"""
    
    session_id = data.get("session_id")
    format_type = data.get("format", "json")  # json or csv
    
    if session_id not in sessions:
        emit("error", {"content": "Session not found"})
        return
    
    session = sessions[session_id]
    dataset = session["state"].get('final_dataset', [])
    
    if format_type == "json":
        content = json.dumps(dataset, indent=2)
        filename = f"dataset_{session_id[:8]}.json"
    else:  # csv
        if not dataset:
            emit("error", {"content": "No data to export"})
            return
        
        csv_path = DATASETS_DIR / f"dataset_{session_id[:8]}.csv"
        
        try:
            # Collect all possible field names from all records
            all_fields = set()
            for record in dataset:
                if isinstance(record, dict):
                    all_fields.update(record.keys())
            
            all_fields = sorted(list(all_fields))
            
            # Ensure all records have all fields (fill missing with empty string)
            normalized_dataset = []
            for record in dataset:
                normalized_record = {}
                for field in all_fields:
                    normalized_record[field] = record.get(field, '') if isinstance(record, dict) else ''
                normalized_dataset.append(normalized_record)
            
            # Write CSV with all fields
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if normalized_dataset:
                    writer = csv.DictWriter(f, fieldnames=all_fields)
                    writer.writeheader()
                    writer.writerows(normalized_dataset)
        except Exception as e:
            emit("error", {"content": f"Error exporting CSV: {e}"})
            return


@socketio.on("get_previous_sessions")
def handle_get_previous_sessions():
    """Get list of previous research sessions for user to continue from"""
    if not session_history:
        emit("previous_sessions", {"sessions": []})
        return
    
    sessions_list = session_history.get_all_sessions()
    emit("previous_sessions", {"sessions": sessions_list})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n{'='*70}")
    print(f"Research Server Starting")
    print(f"   Port: {port}")
    print(f"   URL: http://localhost:{port}")
    print(f"{'='*70}\n")
    
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=True,
        allow_unsafe_werkzeug=True
    )

