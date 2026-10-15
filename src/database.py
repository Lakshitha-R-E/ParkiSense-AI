"""
ParkiSense AI - Database & Authentication Module
=================================================
Lightweight, secure user management using SQLite and PBKDF2-HMAC password hashing.
Zero external dependencies (uses standard library sqlite3 and hashlib).
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Path to database file
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "parkisense.db"


def get_db_connection() -> sqlite3.Connection:
    """Return a connection with Row factory enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using PBKDF2 HMAC-SHA256 with a unique cryptographic salt.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt_bytes = secrets.token_bytes(16)
        salt = salt_bytes.hex()
    else:
        salt_bytes = bytes.fromhex(salt)

    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        iterations=100_000,
    )
    return dk.hex(), salt


def init_db():
    """Initialise SQLite tables and pre-populate default demo accounts and visits."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('Neurologist', 'Patient')),
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                visit_date TEXT NOT NULL,
                updrs_score REAL NOT NULL,
                severity_name TEXT NOT NULL,
                recorded_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(patient_id, visit_date)
            );
        """)

        # Check if users already exist
        cursor.execute("SELECT COUNT(*) AS count FROM users;")
        row = cursor.fetchone()
        if row["count"] == 0:
            # Seed standard demo accounts for effortless evaluation
            demo_accounts = [
                ("dr_smith", "admin123", "Neurologist", "Dr. Sarah Smith (Neurologist)"),
                ("patient_john", "patient123", "Patient", "John Doe (Patient)"),
            ]
            for uname, pwd, role, fname in demo_accounts:
                pwd_hash, salt = hash_password(pwd)
                cursor.execute("""
                    INSERT INTO users (username, password_hash, salt, role, full_name)
                    VALUES (?, ?, ?, ?, ?);
                """, (uname, pwd_hash, salt, role, fname))
            conn.commit()

        # Seed initial visits for demo patients if empty
        cursor.execute("SELECT COUNT(*) AS count FROM patient_visits;")
        v_row = cursor.fetchone()
        if v_row["count"] == 0:
            demo_visits = [
                ("DEMO_001", "John Doe", "2024-01", 22.0, "Mild", "dr_smith"),
                ("DEMO_001", "John Doe", "2024-04", 29.5, "Mild", "dr_smith"),
                ("DEMO_001", "John Doe", "2024-07", 35.1, "Moderate", "dr_smith"),
                ("DEMO_001", "John Doe", "2024-10", 41.8, "Moderate", "dr_smith"),
                ("DEMO_001", "John Doe", "2025-01", 38.2, "Moderate", "dr_smith"),
                ("PT_102", "Alice Wong", "2024-03", 18.5, "Mild", "dr_smith"),
                ("PT_102", "Alice Wong", "2024-09", 24.0, "Mild", "dr_smith"),
                ("PT_102", "Alice Wong", "2025-02", 31.0, "Mild", "dr_smith"),
            ]
            for pid, pname, vdate, score, sev, doc in demo_visits:
                cursor.execute("""
                    INSERT OR IGNORE INTO patient_visits (patient_id, patient_name, visit_date, updrs_score, severity_name, recorded_by)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (pid, pname, vdate, score, sev, doc))
            conn.commit()


def register_user(
    username: str,
    password: str,
    role: str,
    full_name: str,
) -> Tuple[bool, str]:
    """
    Register a new user account.
    Returns (success, message).
    """
    username = username.strip().lower()
    full_name = full_name.strip()

    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not full_name:
        return False, "Full name cannot be empty."
    if role not in ("Neurologist", "Patient"):
        return False, "Role must be 'Neurologist' or 'Patient'."

    pwd_hash, salt = hash_password(password)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, role, full_name)
                VALUES (?, ?, ?, ?, ?);
            """, (username, pwd_hash, salt, role, full_name))
            conn.commit()
            return True, "Account registered successfully! You can now log in."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already registered."
    except Exception as e:
        return False, f"Registration error: {e}"


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Verify user credentials.
    Returns a dict with user info on success, or None on failure.
    """
    username = username.strip().lower()
    if not username or not password:
        return None

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, password_hash, salt, role, full_name
            FROM users
            WHERE username = ?;
        """, (username,))
        row = cursor.fetchone()

        if row is None:
            return None

        pwd_hash, _ = hash_password(password, salt=row["salt"])
        if secrets.compare_digest(pwd_hash, row["password_hash"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "full_name": row["full_name"],
            }
        return None


def add_patient_visit(
    patient_id: str,
    patient_name: str,
    visit_date: str,
    updrs_score: float,
    severity_name: str,
    recorded_by: str = "",
) -> bool:
    """Add or update a patient visit record in the database."""
    patient_id = patient_id.strip()
    patient_name = patient_name.strip()
    visit_date = visit_date.strip()

    if not patient_id or not visit_date:
        return False

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patient_visits (patient_id, patient_name, visit_date, updrs_score, severity_name, recorded_by)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(patient_id, visit_date) DO UPDATE SET
                updrs_score = excluded.updrs_score,
                severity_name = excluded.severity_name,
                patient_name = excluded.patient_name,
                recorded_by = excluded.recorded_by;
        """, (patient_id, patient_name, visit_date, float(updrs_score), severity_name, recorded_by))
        conn.commit()
        return True


def get_patient_visits(patient_id: str) -> List[Dict[str, Any]]:
    """Retrieve all chronological visits for a specific patient."""
    patient_id = patient_id.strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT visit_date, updrs_score, severity_name, patient_name
            FROM patient_visits
            WHERE patient_id = ?
            ORDER BY visit_date ASC;
        """, (patient_id,))
        rows = cursor.fetchall()
        return [
            {
                "date": row["visit_date"],
                "updrs": float(row["updrs_score"]),
                "severity": row["severity_name"],
                "patient_name": row["patient_name"],
            }
            for row in rows
        ]


def get_all_patients() -> List[Dict[str, str]]:
    """
    Retrieve unique patient list across patient_visits and registered users.
    Returns list of dicts: [{'patient_id': ..., 'patient_name': ...}].
    """
    patients_dict = {}

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # From visits table
        cursor.execute("SELECT DISTINCT patient_id, patient_name FROM patient_visits;")
        for row in cursor.fetchall():
            patients_dict[row["patient_id"]] = row["patient_name"]

        # From registered users with Patient role
        cursor.execute("SELECT username, full_name FROM users WHERE role = 'Patient';")
        for row in cursor.fetchall():
            if row["username"] not in patients_dict:
                patients_dict[row["username"]] = row["full_name"]

    # Sort by patient_id
    sorted_patients = sorted(patients_dict.items(), key=lambda x: x[0])
    return [{"patient_id": pid, "patient_name": pname} for pid, pname in sorted_patients]

