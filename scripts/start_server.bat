@echo off
title CryptoGuard GNN - AML & Graph Forensics Command Center
cd /d "%~dp0.."
echo Starting CryptoGuard GNN FastAPI Command Center on http://localhost:8000 ...
start http://localhost:8000
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
