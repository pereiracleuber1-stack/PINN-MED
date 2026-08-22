import sqlite3
import hashlib
import os
import datetime

class AuthManager:
    """
    Gerenciador de Usuários, Senhas e Perfis RBAC
    Conformidade 21 CFR Part 11 / NIST SP 800-132.
    """
    def __init__(self, db_path="audit_ledger.db"):
        self.db_path = db_path
        self._init_db()
        self._seed_default_admin()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                crm_registration TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _hash_password(self, password: str, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return pwd_hash.hex(), salt.hex()

    def _seed_default_admin(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            pwd_hash, salt_hex = self._hash_password("pinn2026")
            ts = datetime.datetime.utcnow().isoformat()
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, crm_registration, role, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("admin", pwd_hash, salt_hex, "Dr. Cleuber Pereira Ramos", "CRM-MT 10452", "Administrador de Sistema", "ACTIVE", ts))
            conn.commit()
        conn.close()

    def register_user(self, username, password, full_name, crm, role):
        username = username.strip().lower()
        if not username or not password:
            return False, "Usuário e senha são obrigatórios."
        
        pwd_hash, salt_hex = self._hash_password(password)
        ts = datetime.datetime.utcnow().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, crm_registration, role, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (username, pwd_hash, salt_hex, full_name, crm, role, "ACTIVE", ts))
            conn.commit()
            conn.close()
            return True, "Usuário cadastrado com sucesso!"
        except sqlite3.IntegrityError:
            return False, f"O nome de usuário '{username}' já existe."

    def authenticate_user(self, username, password):
        username = username.strip().lower()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, salt, full_name, crm_registration, role, status FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False, None
        
        stored_hash, salt_hex, full_name, crm, role, status = row
        if status != "ACTIVE":
            return False, "Conta desativada pelo administrador."

        salt = bytes.fromhex(salt_hex)
        calc_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000).hex()
        
        if calc_hash == stored_hash:
            return True, {
                "username": username,
                "full_name": full_name,
                "crm": crm,
                "role": role
            }
        return False, None

    def change_password(self, username, old_password, new_password):
        auth_ok, user_data = self.authenticate_user(username, old_password)
        if not auth_ok:
            return False, "Senha atual incorreta."
        
        pwd_hash, salt_hex = self._hash_password(new_password)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?", (pwd_hash, salt_hex, username.lower()))
        conn.commit()
        conn.close()
        return True, "Senha alterada com sucesso!"

    def list_all_users(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, crm_registration, role, status, created_at FROM users ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()
        return rows

    def set_user_status(self, username, new_status):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET status = ? WHERE username = ?", (new_status, username.lower()))
        conn.commit()
        conn.close()
        return True
