#!/usr/bin/env bash
# Daily PostgreSQL backup for AlphaVedha.
# Usage: ./scripts/backup_db.sh [backup_dir]
# Cron:  0 2 * * * /opt/alphavedha/scripts/backup_db.sh /opt/alphavedha/backups
set -euo pipefail

BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"

DB_HOST="${PGHOST:-localhost}"
DB_PORT="${PGPORT:-5435}"
DB_USER="${POSTGRES_USER:-alphavedha}"
DB_NAME="${POSTGRES_DB:-alphavedha}"

mkdir -p "$BACKUP_DIR"

BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_${TIMESTAMP}.sql.gz"

echo "[$(date -Iseconds)] Starting backup: $DB_NAME -> $BACKUP_FILE"

pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --no-owner \
    --no-privileges \
    --format=plain \
    | gzip > "$BACKUP_FILE"

FILESIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[$(date -Iseconds)] Backup complete: $FILESIZE"

# Verify backup is not empty
if [ ! -s "$BACKUP_FILE" ]; then
    echo "ERROR: Backup file is empty!" >&2
    rm -f "$BACKUP_FILE"
    exit 1
fi

# Prune old backups
DELETED=$(find "$BACKUP_DIR" -name "${DB_NAME}_*.sql.gz" -mtime +"$KEEP_DAYS" -delete -print | wc -l)
if [ "$DELETED" -gt 0 ]; then
    echo "[$(date -Iseconds)] Pruned $DELETED backup(s) older than $KEEP_DAYS days"
fi

echo "[$(date -Iseconds)] Done. Active backups:"
ls -lh "$BACKUP_DIR"/${DB_NAME}_*.sql.gz 2>/dev/null || echo "  (none)"
