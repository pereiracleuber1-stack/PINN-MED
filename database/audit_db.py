import sqlite3
import hashlib
import json
from datetime import datetime

DB_PATH = "database/pinn_audit.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            operator TEXT,
            patient_id TEXT,
            action TEXT,
            model_type TEXT,
            params_json TEXT,
            risk_level TEXT,
            record_hash TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(operator, patient_id, action, model_type, params_dict, risk_level):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    ts = datetime.utcnow().isoformat() + "Z"
    params_str = json.dumps(params_dict, sort_keys=True)
    
    # Hash SHA-256 para integridade do registro médico
    raw_payload = f"{ts}|{operator}|{patient_id}|{action}|{model_type}|{params_str}|{risk_level}"
    record_hash = hashlib.sha256(raw_payload.encode()).hexdigest()
    
    cursor.execute("""
        INSERT INTO audit_logs (timestamp, operator, patient_id, action, model_type, params_json, risk_level, record_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, operator, patient_id, action, model_type, params_str, risk_level, record_hash))
    conn.commit()
    conn.close()
    return record_hash

def get_all_logs(limit=100):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, operator, patient_id, action, model_type, risk_level, record_hash FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows
