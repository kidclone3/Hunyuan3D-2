import argparse
import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


TOKEN_PREFIX = "h3d"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _connect(database_path):
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS api_credentials (
            token_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            salt TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            revoked_at TEXT
        )
        """
    )
    return connection


def _token_hash(secret, salt):
    return hashlib.sha256(bytes.fromhex(salt) + secret.encode("utf-8")).hexdigest()


def create_credential(database_path, name):
    name = name.strip()
    if not name:
        raise ValueError("Credential name cannot be empty")
    token_id = secrets.token_hex(8)
    secret = secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    token_hash = _token_hash(secret, salt)
    with _connect(database_path) as connection:
        connection.execute(
            "INSERT INTO api_credentials "
            "(token_id, name, salt, token_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (token_id, name, salt, token_hash, _utc_now()),
        )
    return token_id, f"{TOKEN_PREFIX}_{token_id}_{secret}"


def verify_credential(database_path, token):
    try:
        prefix, token_id, secret = token.split("_", 2)
    except ValueError:
        return False
    if prefix != TOKEN_PREFIX or not token_id or not secret:
        return False
    with _connect(database_path) as connection:
        row = connection.execute(
            "SELECT salt, token_hash FROM api_credentials "
            "WHERE token_id = ? AND revoked_at IS NULL",
            (token_id,),
        ).fetchone()
        if row is None or not secrets.compare_digest(_token_hash(secret, row[0]), row[1]):
            return False
        connection.execute(
            "UPDATE api_credentials SET last_used_at = ? WHERE token_id = ?",
            (_utc_now(), token_id),
        )
    return True


def list_credentials(database_path):
    with _connect(database_path) as connection:
        return connection.execute(
            "SELECT token_id, name, created_at, last_used_at, revoked_at "
            "FROM api_credentials ORDER BY created_at"
        ).fetchall()


def revoke_credential(database_path, token_id):
    with _connect(database_path) as connection:
        cursor = connection.execute(
            "UPDATE api_credentials SET revoked_at = ? "
            "WHERE token_id = ? AND revoked_at IS NULL",
            (_utc_now(), token_id),
        )
    return cursor.rowcount == 1


def main():
    parser = argparse.ArgumentParser(description="Manage Hunyuan3D API credentials")
    parser.add_argument("--database", default="data/api_keys.sqlite3")
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create", help="Generate a credential")
    create_parser.add_argument("--name", required=True)
    commands.add_parser("list", help="List credentials without revealing secrets")
    revoke_parser = commands.add_parser("revoke", help="Revoke a credential")
    revoke_parser.add_argument("token_id")
    args = parser.parse_args()

    if args.command == "create":
        token_id, token = create_credential(args.database, args.name)
        print(f"Credential ID: {token_id}")
        print(f"Credential: {token}")
        print("Save this credential now; it cannot be displayed again.")
    elif args.command == "list":
        print("ID\tNAME\tCREATED\tLAST USED\tREVOKED")
        for row in list_credentials(args.database):
            print("\t".join(value or "-" for value in row))
    elif not revoke_credential(args.database, args.token_id):
        parser.error("Credential was not found or is already revoked")
    else:
        print(f"Revoked credential {args.token_id}")


if __name__ == "__main__":
    main()
