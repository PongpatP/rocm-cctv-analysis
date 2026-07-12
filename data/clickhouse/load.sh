#!/usr/bin/env bash
# Recreate the schema, then load the carried tables. Run from the deployment root
# once ClickHouse is healthy:  bash data/clickhouse/load.sh
set -e
CH="docker compose exec -T clickhouse clickhouse-client"
D="$(cd "$(dirname "$0")" && pwd)"
$CH --multiquery < "$D/01_schema.sql"
for f in "$D"/*.native; do
  t=$(basename "$f" .native); t=${t%.sample_2h}
  echo "loading $t"
  $CH -q "INSERT INTO ccvt.$t FORMAT Native" < "$f"
done
echo "done"
