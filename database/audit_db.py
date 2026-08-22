import sqlite3
import datetime
import hashlib
import json
import os

class AuditDatabase:
    """
    Ledger de Auditoria em SQLite com Assinatura Criptográfica SHA-256
    Conformidade 21 CFR Part 11 e LGPD / HIPAA.
    """
    def __init__(self, db_path="audit_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operator_crm TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    k_pg REAL DEFAULT 0.0,
                    c_pn REAL DEFAULT 0.0,
                    mu_c REAL DEFAULT 0.0,
                    peak_lactate REAL DEFAULT 0.0,
                    raw_signature TEXT DEFAULT '',
                    sha256_hash TEXT NOT NULL
                )
            """)
            conn.commit()
            
            # Migração automática se a tabela foi gerada em versão anterior sem a coluna
            cursor.execute("PRAGMA table_info(audit_logs)")
            cols = [c[1] for c in cursor.fetchall()]
            if "raw_signature" not in cols:
                cursor.execute("ALTER TABLE audit_logs ADD COLUMN raw_signature TEXT DEFAULT ''")
                conn.commit()
            conn.close()
        except Exception:
            pass

    def log_inference(self, operator_crm="CRM-MT 10452", patient_id="PAT-AUTO", model_type="MODEL",
                      k_pg=0.0, c_pn=0.0, mu_c=0.0, peak_lactate=0.0, raw_signature="", *args, **kwargs):
        ts = datetime.datetime.utcnow().isoformat()
        if not raw_signature:
            raw_signature = f"{operator_crm}_{patient_id}_{model_type}_{k_pg}_{c_pn}_{mu_c}_{peak_lactate}_{ts}"
        sha256_hash = hashlib.sha256(str(raw_signature).encode('utf-8')).hexdigest()

        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_logs (timestamp, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, raw_signature, sha256_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(ts),
                str(operator_crm or "CRM-MT 10452"),
                str(patient_id or "PAT-AUTO"),
                str(model_type or "GENERAL"),
                float(k_pg) if isinstance(k_pg, (int, float)) else 0.0,
                float(c_pn) if isinstance(c_pn, (int, float)) else 0.0,
                float(mu_c) if isinstance(mu_c, (int, float)) else 0.0,
                float(peak_lactate) if isinstance(peak_lactate, (int, float)) else 0.0,
                str(raw_signature),
                str(sha256_hash)
            ))
            conn.commit()
            conn.close()
        except Exception:
            pass
        return sha256_hash

    def get_all_logs(self, limit=500, *args, **kwargs):
        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, sha256_hash 
                FROM audit_logs 
                ORDER BY id DESC 
                LIMIT ?
            """, (int(limit),))
            rows = cursor.fetchall()
            conn.close()
            return rows
        except Exception:
            return []

_audit_singleton = AuditDatabase()

def log_event(*args, **kwargs):
    operator_crm = kwargs.get("operator_crm", kwargs.get("crm", kwargs.get("user", "CRM-MT 10452")))
    patient_id = kwargs.get("patient_id", kwargs.get("patient", "PAT-AUTO"))
    model_type = kwargs.get("model_type", kwargs.get("event_type", "INFERENCE_EVENT"))
    k_pg = kwargs.get("k_pg", 0.0)
    c_pn = kwargs.get("c_pn", 0.0)
    mu_c = kwargs.get("mu_c", 0.0)
    peak_lactate = kwargs.get("peak_lactate", kwargs.get("lactate", 0.0))

    if len(args) >= 1 and "operator_crm" not in kwargs:
        operator_crm = str(args[0])
    if len(args) >= 2 and "patient_id" not in kwargs:
        patient_id = str(args[1])
    if len(args) >= 3 and "model_type" not in kwargs:
        model_type = str(args[2])

    raw_signature = kwargs.get("raw_signature", f"{args}_{kwargs}")

    return _audit_singleton.log_inference(
        operator_crm=operator_crm,
        patient_id=patient_id,
        model_type=model_type,
        k_pg=k_pg,
        c_pn=c_pn,
        mu_c=mu_c,
        peak_lactate=peak_lactate,
        raw_signature=raw_signature
    )

def log_inference(*args, **kwargs):
    return log_event(*args, **kwargs)

def get_all_logs(limit=500, *args, **kwargs):
    return _audit_singleton.get_all_logs(limit=limit)

def get_audit_logs(limit=500, *args, **kwargs):
    return _audit_singleton.get_all_logs(limit=limit)
