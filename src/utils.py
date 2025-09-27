# src/utils.py
import os
from dotenv import load_dotenv
from pathlib import Path
import sqlite3
import json

load_dotenv()

def env(k, default=None):
    return os.environ.get(k, default)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def init_metadata_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS docs (
            id TEXT PRIMARY KEY,
            source TEXT,
            metadata TEXT
        )
    ''')
    conn.commit()
    return conn

def store_metadata(conn, doc_id, source, metadata: dict):
    cur = conn.cursor()
    cur.execute('REPLACE INTO docs (id, source, metadata) VALUES (?, ?, ?)',
                (doc_id, source, json.dumps(metadata)))
    conn.commit()

def load_metadata(conn, doc_id):
    cur = conn.cursor()
    cur.execute('SELECT metadata FROM docs WHERE id=?', (doc_id,))
    row = cur.fetchone()
    return json.loads(row[0]) if row else None
