#!/bin/bash

echo "Starting backend..."
python endpoints.py --model="datamuse" &    # Could also be set to "ngram"

echo "Starting frontend..."
cd frontend
npm install
npm run dev &

sleep 5
echo "Opening browser..."
open http://localhost:5173

echo "Services running - Press Ctrl+C to stop"
wait
