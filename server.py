"""
server.py
---------
Uvicorn entry point for the Arch-Agent FastAPI server.

Usage:
    python server.py
    # or directly:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
