#!/usr/bin/env bash
# create_venv.sh
set -e

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}

echo "Using python: $(which $PYTHON)"

# Create venv
if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv $VENV_DIR exists. Activating..."
else
  echo "Creating virtualenv in $VENV_DIR..."
  $PYTHON -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo "Copy env.example -> .env and edit if needed."
if [ ! -f .env ]; then
  cp env.example .env
  echo ".env file created from env.example — please edit .env to set OPENAI_API_KEY or use local LLM."
fi

echo "Done. Activate with: source $VENV_DIR/bin/activate"
