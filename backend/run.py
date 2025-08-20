#!/usr/bin/env python3
"""
Simple startup script for the Semantic Product Search API.
Run this file to start the FastAPI server.
"""

import uvicorn
from app.main import app

if __name__ == "__main__":
    print("Starting Semantic Product Search API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


