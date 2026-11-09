#!/bin/zsh
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
