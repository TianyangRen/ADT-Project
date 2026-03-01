#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="."
streamlit run ui/app.py
