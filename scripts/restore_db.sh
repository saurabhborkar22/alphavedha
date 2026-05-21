#!/usr/bin/env bash
# Restore AlphaVedha database from a backup file.
# Usage: ./scripts/restore_db.sh backups/alphavedha_20260521_020000.sql.gz
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <backup_file.sql.gz>" >&2
    echo "Available backups:" >&2
    ls -lh backups/*.sql.gz 2>/dev/null || echo "  (none found in ./backups/)" >&2
    exit 1
fi

BACKUP_FILE="$1"
DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5435}"
DB_USER="${POSTGRES_USER:-alphavedha}"
DB_NAME="${POSTGRES_DB:-alphavedha}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE" >&2
    exit 1
fi

echo "WARNING: This will DROP and recreate the database '$DB_NAME'."
echo "Backup file: $BACKUP_FILE"
read -rp "Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo "[$(date -Iseconds)] Dropping and recreating database..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "
    SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '$DB_NAME' AND pid <> pg_backend_pid();
" 2>/dev/null || true
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $DB_NAME;"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $DB_NAME;"

echo "[$(date -Iseconds)] Restoring from backup..."
gunzip -c "$BACKUP_FILE" | psql \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --single-transaction \
    -q \
    2>&1 || true

echo "[$(date -Iseconds)] Running Alembic stamp to mark current migration..."
alembic stamp head

echo "[$(date -Iseconds)] Restore complete. Verifying table counts..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
    SELECT schemaname, tablename, n_live_tup as row_count
    FROM pg_stat_user_tables
    ORDER BY n_live_tup DESC;
"
