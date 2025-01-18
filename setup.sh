#!/usr/bin/env bash

if [ ! -d "$(dirname $0)/.venv" ]; then
	echo "no venv found, creating venv for llm agents project..."
	python -m venv "$(dirname $0)/.venv"
fi

source "$(dirname $0)/.venv/bin/activate"
echo "activated venv for llm agents project"

echo "ensuring pip exists in the environment"
python -m ensurepip

echo "installing required dependencies"
pip install -r "$(dirname $0)/requirements.txt"
