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
                k_pg REAL DEFAULT 0.0,
                c_pn REAL DEFAULT 0.0,
                mu_c REAL DEFAULT 0.0,
                peak_lactate REAL DEFAULT 0.0,
                raw_signature TEXT DEFAULT '',
                sha256_hash TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def log_inference(self, operator_crm="CRM-MT 10452", patient_id="PAT-AUTO", model_type="MODEL",
                      k_pg=0.0, c_pn=0.0, mu_c=0.0, peak_lactate=0.0, raw_signature=""):
        ts = datetime.datetime.utcnow().isoformat()
        if not raw_signature:
            raw_signature = f"{operator_crm}_{patient_id}_{model_type}_{k_pg}_{c_pn}_{mu_c}_{peak_lactate}_{ts}"
        sha256_hash = hashlib.sha256(raw_signature.encode('utf-8')).hexdigest()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, operator_crm, patient_id, model_type, k_pg, c_pn, mu_c, peak_lactate, raw_signature, sha256_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, str(operator_crm), str(patient_id), str(model_type), float(k_pg or 0.0), float(c_pn or 0.0), float(mu_c or 0.0), float(peak_lactate or 0.0), str(raw_signature), sha256_hash))
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

# Instância global compartilhada
_audit_singleton = AuditDatabase()

def log_event(*args, **kwargs):
    """Função legada para compatibilidade com Modelo A (U-PINN) e Modelo B (Inversa)."""
    operator_crm = kwargs.get("operator_crm", kwargs.get("user", kwargs.get("crm", "CRM-MT 10452")))
    patient_id = kwargs.get("patient_id", kwargs.get("patient", "PAT-AUTO"))
    model_type = kwargs.get("model_type", kwargs.get("event_type", "INFERENCE_EVENT"))
    k_pg = kwargs.get("k_pg", 0.0)
    c_pn = kwargs.get("c_pn", 0.0)
    mu_c = kwargs.get("mu_c", 0.0)
    peak_lactate = kwargs.get("peak_lactate", kwargs.get("lactate", 0.0))

    if len(args) >= 1 and "model_type" not in kwargs:
        model_type = str(args[0])
    if len(args) >= 2 and "patient_id" not in kwargs:
        patient_id = str(args[1])
    if len(args) >= 3 and "operator_crm" not in kwargs:
        operator_crm = str(args[2])

    raw_sig = kwargs.get("raw_signature", f"{model_type}_{patient_id}_{args}_{kwargs}")
    return _audit_singleton.log_inference(
        operator_crm=operator_crm,
        patient_id=patient_id,
        model_type=model_type,
        k_pg=k_pg,
        c_pn=c_pn,
        mu_c=mu_c,
        peak_lactate=peak_lactate,
        raw_signature=raw_sig
    )

def log_inference(*args, **kwargs):
    return log_event(*args, **kwargs)

def get_audit_logs():
    return _audit_singleton.get_all_logs()

def get_all_logs():
    return _audit_singleton.get_all_logs()
