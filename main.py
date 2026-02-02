#!/usr/bin/env python3
"""
Main entry point for the Automated Research System.
Starts the Flask WebSocket server for dataset building.
"""

import os
import sys
import argparse
from dotenv import find_dotenv, load_dotenv

# Load environment variables
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)

from server import app, socketio


def main():
    """Start the Flask server"""
    
    parser = argparse.ArgumentParser(
        description="Automated Research System - Dataset Builder"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("FLASK_HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FLASK_PORT", 5000)),
        help="Server port (default: 5000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("FLASK_DEBUG", "False").lower() == "true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Print startup info
    print("\n" + "="*70)
    print("Automated Research System - Dataset Builder")
    print("="*70)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Debug Mode: {args.debug}")
    print(f"OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-5-mini')}")
    print(f"SEARXNG URL: {os.getenv('SEARXNG_URL', 'http://localhost:8080')}")
    print("="*70)
    print("Open browser to the server URL above to get started.")
    print("="*70 + "\n")
    
    try:
        # Start the SocketIO server with Flask
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True
        )
    except KeyboardInterrupt:
        print("\n\n✋ Server shutdown requested")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()