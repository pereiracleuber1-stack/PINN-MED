import numpy as np
import time

def fetch_telemetry_tick(current_hour, seed_id=42):
    """Simula a ingestão contínua de pacotes de dados de um leito hospitalar."""
    np.random.seed(int(time.time() * 1000) % 100000 + seed_id)
    noise = np.random.normal(0, 0.05)
    
    pam_base = 82.0 - (18.0 / (1.0 + np.exp(-(current_hour - 16) / 3.5)))
    pam = max(40.0, pam_base + np.random.normal(0, 1.2))
    
    lactato_base = 1.1 + 0.04 * current_hour + 0.003 * (current_hour ** 1.8)
    lactato = max(0.5, lactato_base + noise)
    
    fc = 75.0 + 1.2 * current_hour + np.random.normal(0, 2.0)
    spo2 = max(88.0, 98.0 - 0.15 * current_hour + np.random.normal(0, 0.5))
    
    return {
        "hour": current_hour,
        "pam": round(pam, 1),
        "lactato": round(lactato, 2),
        "fc": round(fc, 1),
        "spo2": round(spo2, 1)
    }
