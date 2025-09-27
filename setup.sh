#!/bin/bash
echo "Creating virtual env..."
conda create -n finagent python=3.11 -y
conda activate finagent

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Neo4j container..."
docker run -d --name neo4j-finagent \
   -p7474:7474 -p7687:7687 \
   -v $HOME/neo4j/data:/data \
   neo4j:latest


pip install sec-edgar-downloader
