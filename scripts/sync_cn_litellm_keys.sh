#!/bin/bash
# Copy virtual keys from GPU4's LiteLLM database into the A100's local one.
# GPU4 is the source of truth, so change keys there. Spend stays per region and is never overwritten.
# A failed run changes nothing and the A100 keeps serving from its last good copy.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")"

SRC=$(grep -E '^LITELLM_SYNC_SOURCE_URL=' gateway.env | cut -d= -f2-)
PASS=$(cut -d= -f2- pg-local.env)

sudo -n docker exec -i -e SRC="$SRC" -e PGPASSWORD="$PASS" -e PGCONNECT_TIMEOUT=30 llm-litellm-pg bash -s <<'INNER'
set -euo pipefail
OPTS="with (format csv, quote e'\\x01', delimiter e'\\x02')"
LOCAL=(-h 127.0.0.1 -p 15433 -U litellm -d litellm -X -q -v ON_ERROR_STOP=1)
# Only keys: no key uses LiteLLM's budget, organization or project tables. A key that starts to will fail the
# foreign key, and the sync then changes nothing until that table is added here.
TABLES=(
  LiteLLM_VerificationToken:token
)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# One session for all tables: each new connection costs several round trips over the border crossing.
for entry in "${TABLES[@]}"; do
  table=${entry%%:*}
  echo "\\copy (select to_jsonb(x)::text from \"$table\" x) to '$T/$table' $OPTS"
done | psql "$SRC" -X -q -v ON_ERROR_STOP=1

{
  echo 'create temp table incoming (tbl text, doc jsonb) on commit drop;'
  for entry in "${TABLES[@]}"; do
    table=${entry%%:*}
    echo "\\copy incoming (doc) from '$T/$table' $OPTS"
    echo "update incoming set tbl = '$table' where tbl is null;"
  done
  cat <<SQL
do \$sync\$
declare
  entry text;
  tbl_name text;
  pk text;
  assignments text;
  removed bigint;
begin
  if not exists (select 1 from incoming where tbl = 'LiteLLM_VerificationToken') then
    raise exception 'source returned no keys, refusing to sync';
  end if;
  foreach entry in array array[$(printf "'%s'," "${TABLES[@]}" | sed 's/,$//')] loop
    tbl_name := split_part(entry, ':', 1);
    pk := split_part(entry, ':', 2);
    select string_agg(format('%I = excluded.%I', column_name, column_name), ', ') into assignments
    from information_schema.columns
    where table_schema = 'public' and table_name = tbl_name
      and column_name <> all (array[pk, 'spend', 'model_spend', 'last_active']);
    execute format(
      'insert into %I select r.* from incoming i, jsonb_populate_record(null::%I, i.doc) r where i.tbl = %L '
      'on conflict (%I) do update set %s',
      tbl_name, tbl_name, tbl_name, pk, assignments
    );
  end loop;
  delete from "LiteLLM_VerificationToken"
  where token not in (select doc->>'token' from incoming where tbl = 'LiteLLM_VerificationToken');
  get diagnostics removed = row_count;
  raise notice 'synced keys=% removed=%',
    (select count(*) from incoming where tbl = 'LiteLLM_VerificationToken'), removed;
end
\$sync\$;
SQL
} | psql "${LOCAL[@]}" --single-transaction
INNER
