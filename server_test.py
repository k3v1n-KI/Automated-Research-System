import os
import json
import asyncio
from datetime import datetime
from uuid import uuid4

from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit, disconnect
from openai import AsyncOpenAI
from dotenv import find_dotenv, load_dotenv

# Load environment
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
# Flask app with template support
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "test-secret")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Store conversation history per session
conversations = {}


@app.route("/")
def index():
    """Serve chat interface"""
    return render_template('chat_interface.html')


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(conversations)
    }


@socketio.on("connect")
def handle_connect():
    """Handle client connection"""
    print(f"✓ Client connected: {request.sid}")
    emit("connected", {"timestamp": datetime.now().isoformat()})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection"""
    print(f"✓ Client disconnected: {request.sid}")
    # Clean up conversation if exists
    if request.sid in conversations:
        del conversations[request.sid]


@socketio.on("new_session")
def handle_new_session(data):
    """Start a new conversation session"""
    session_id = data.get("session_id", str(uuid4()))
    
    # Initialize conversation history
    conversations[request.sid] = {
        "session_id": session_id,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    
    print(f"New session: {session_id} (client: {request.sid})")
    emit("session_created", {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    })


@socketio.on("message")
def handle_message(data):
    """Handle chat message and stream response"""
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id")
    
    if not user_message:
        emit("error", {"content": "Empty message"})
        return
    
    if request.sid not in conversations:
        emit("error", {"content": "No active session. Send 'new_session' first."})
        return
    
    print(f"Message from {request.sid}: {user_message}")
    
    # Run chat in background with proper async handling
    socketio.start_background_task(
        _chat_task_wrapper,
        user_message,
        request.sid,
        session_id
    )


def _chat_task_wrapper(user_message: str, client_id: str, session_id: str):
    """Wrapper to run async chat task"""
    try:
        asyncio.run(_chat_task(user_message, client_id, session_id))
    except RuntimeError:
        # Event loop already exists, use it
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop in this thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(asyncio.run, _chat_task(user_message, client_id, session_id))
        else:
            loop.run_until_complete(_chat_task(user_message, client_id, session_id))


async def _chat_task(user_message: str, client_id: str, session_id: str):
    """Chat task that streams response"""
    
    try:
        # Get conversation history
        conv = conversations.get(client_id)
        if not conv:
            socketio.emit("error", {"content": "Session lost"}, room=client_id)
            return
        
        messages = conv["messages"]
        
        # Add user message to history
        messages.append({"role": "user", "content": user_message})
        
        # Stream LLM response
        socketio.emit("message_start", {
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        }, room=client_id)
        
        full_response = ""
        
        stream = await openai_client.chat.completions.create(
            model=openai_model,
            messages=messages,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                
                # Emit token to client
                socketio.emit("token", {
                    "token": token,
                    "timestamp": datetime.now().isoformat()
                }, room=client_id)
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": full_response})
        
        # Emit completion
        socketio.emit("message_complete", {
            "role": "assistant",
            "content": full_response,
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat()
        }, room=client_id)
        
        print(f"Response complete for {client_id} ({len(full_response)} chars)")
        
    except Exception as e:
        print(f"Error in chat task: {e}")
        socketio.emit("error", {"content": str(e)}, room=client_id)


@socketio.on("get_history")
def handle_get_history(data):
    """Get conversation history"""
    session_id = data.get("session_id")
    
    if request.sid not in conversations:
        emit("error", {"content": "No active session"})
        return
    
    conv = conversations[request.sid]
    emit("history", {
        "session_id": conv["session_id"],
        "messages": conv["messages"],
        "message_count": len(conv["messages"]),
        "created_at": conv["created_at"]
    })


@socketio.on("clear_history")
def handle_clear_history(data):
    """Clear conversation history"""
    if request.sid not in conversations:
        emit("error", {"content": "No active session"})
        return
    
    conversations[request.sid]["messages"] = []
    emit("history_cleared", {
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting test server on port {port}...")
    print("Events:")
    print("  - 'new_session': Start a new conversation")
    print("  - 'message': Send a chat message")
    print("  - 'get_history': Retrieve conversation history")
    print("  - 'clear_history': Clear conversation history")
    print()
    
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=os.getenv("DEBUG", "False").lower() == "true",
        allow_unsafe_werkzeug=True
    )
