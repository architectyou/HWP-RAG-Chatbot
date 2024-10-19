#!/bin/bash
ollama pull llama3.1
ollama serve &

sleep 10


chainlit run main.py