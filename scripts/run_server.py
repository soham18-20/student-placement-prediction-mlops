"""
Script: scripts/run_server.py
Starts the Flask development server for the Student Placement Prediction System.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app

def main():
    parser = argparse.ArgumentParser(description="Run Student Placement Prediction System Flask Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    app = create_app()
    print(f"\n=======================================================")
    print(f"  Student Placement Prediction System with MLOps")
    print(f"  Dashboard UI  : http://{args.host}:{args.port}/")
    print(f"  Health Check  : http://{args.host}:{args.port}/health")
    print(f"  Prediction API: http://{args.host}:{args.port}/api/predict")
    print(f"=======================================================\n")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
