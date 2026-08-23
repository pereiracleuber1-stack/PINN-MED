import sqlite3
import hashlib
import os
import datetime
import json

class AuthManager:
    """
    Gerenciador de Usuários, Senhas, Perfis RBAC e Licenciamento Modular de Motores.
    """
    def __init__(self, db_path="audit_ledger.db"):
        self.db_path = db_path
        self._init_db()
        self._seed_default_admin()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                registration_id TEXT NOT NULL,
                role TEXT NOT NULL,
                allowed_modules TEXT NOT NULL DEFAULT 'ALL',
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

        # Migração automática para suporte a módulos permitidos
        cursor.execute("PRAGMA table_info(users)")
        cols = [c[1] for c in cursor.fetchall()]
        if "allowed_modules" not in cols:
            cursor.execute("ALTER TABLE users ADD COLUMN allowed_modules TEXT NOT NULL DEFAULT 'ALL'")
            conn.commit()
        if "registration_id" not in cols and "crm_registration" in cols:
            cursor.execute("ALTER TABLE users RENAME COLUMN crm_registration TO registration_id")
            conn.commit()
        conn.close()

    def _hash_password(self, password: str, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return pwd_hash.hex(), salt.hex()

    def _seed_default_admin(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            pwd_hash, salt_hex = self._hash_password("pinn2026")
            ts = datetime.datetime.utcnow().isoformat()
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, registration_id, role, allowed_modules, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "admin", pwd_hash, salt_hex,
                "Prof. Me. Cleuber Pereira Ramos",
                "PROFMAT / Pesquisador em Matemática Computacional",
                "Pesquisador Chefe & Administrador",
                json.dumps(["ALL"]),
                "ACTIVE", ts
            ))
            conn.commit()
        conn.close()

    def register_user(self, username, password, full_name, reg_id, role, allowed_modules=None):
        username = username.strip().lower()
        if not username or not password:
            return False, "Usuário e senha são obrigatórios."
        
        modules_json = json.dumps(allowed_modules if allowed_modules else ["ALL"])
        pwd_hash, salt_hex = self._hash_password(password)
        ts = datetime.datetime.utcnow().isoformat()
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, full_name, registration_id, role, allowed_modules, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (username, pwd_hash, salt_hex, full_name, reg_id, role, modules_json, "ACTIVE", ts))
            conn.commit()
            conn.close()
            return True, "Operador registrado com sucesso!"
        except sqlite3.IntegrityError:
            return False, f"O usuário '{username}' já existe."

    def authenticate_user(self, username, password):
        username = username.strip().lower()
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, salt, full_name, registration_id, role, allowed_modules, status FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False, None
        
        stored_hash, salt_hex, full_name, reg_id, role, mod_json, status = row
        if status != "ACTIVE":
            return False, "Conta inativa."

        salt = bytes.fromhex(salt_hex)
        calc_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000).hex()
        
        if calc_hash == stored_hash:
            try:
                allowed_mods = json.loads(mod_json)
            except Exception:
                allowed_mods = ["ALL"]
            return True, {
                "username": username,
                "full_name": full_name,
                "reg_id": reg_id,
                "role": role,
                "allowed_modules": allowed_mods
            }
        return False, None

    def change_password(self, username, old_password, new_password):
        auth_ok, _ = self.authenticate_user(username, old_password)
        if not auth_ok:
            return False, "Senha atual incorreta."
        pwd_hash, salt_hex = self._hash_password(new_password)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE username = ?", (pwd_hash, salt_hex, username.lower()))
        conn.commit()
        conn.close()
        return True, "Senha atualizada com sucesso!"

    def list_all_users(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, registration_id, role, allowed_modules, status, created_at FROM users ORDER BY id ASC")
        rows = cursor.fetchall()
        conn.close()
        return rows
