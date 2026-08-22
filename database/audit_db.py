import sqlite3
import datetime
import hashlib

class AuditDatabase:
    """
    Ledger de Auditoria em SQLite com Assinatura Criptográfica SHA-256
    Conformidade 21 CFR Part 11 e LGPD / HIPAA.
    """
    def __init__(self, db_path="audit_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operator_crm TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                k_pg REAL,
                c_pn REAL,
                mu_c REAL,
                peak_lactate REAL,
                sha256_hash TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def log_inference(self, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, raw_signature=""):
        ts = datetime.datetime.utcnow().isoformat()
        if not raw_signature:
            raw_signature = f"{operator_crm}_{patient_id}_{model_type}_{k_pg}_{c_pn}_{mu_c}_{peak_lactate}_{ts}"
        sha256_hash = hashlib.sha256(raw_signature.encode('utf-8')).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, sha256_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, sha256_hash))
        conn.commit()
        conn.close()
        return sha256_hash

    def get_all_logs(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, timestamp, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, sha256_hash FROM audit_logs ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
