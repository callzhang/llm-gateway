# Other callers' requests in the spend logs (retry breadcrumbs)

## Cause

LiteLLM 1.99.0 `Router.log_retry` appends each failed attempt's kwargs (Responses `input`, and
`proxy_server_request.body` with all messages and images) to `self.previous_models`, one list per
Router, and attaches that list to the current request's `metadata` / `litellm_metadata`.
`proxy_server_request.body` holds that same dict, so it was stored in `LiteLLM_SpendLogs`.
The order 1 -> order 2 overflow fallback calls `log_retry`, so `num_retries: 0` did not prevent it.

Upstream: BerriAI/litellm issue 24965, fixed by PR 39491 (1.101.0, backported to 1.100.1).
1.100.0 is worse (breadcrumbs grow without bound, OOM); do not upgrade to it.

Fix here: `retry_breadcrumb_hook.py`. It takes effect at the next LiteLLM restart in each region.

## Stored rows

Measured read-only, 2026-11-26.

### GPU4

The first row holding another key's data is from 2026-11-23.

| | |
|---|---|
| Spend-log rows | 123,531 |
| Rows containing `previous_models` | 8,312 (7,655 success, 657 failure) |
| ... holding another virtual key's hash (a lower bound for cross-key content) | 744 |
| ... of those, also holding an inline image | 13 |
| `memory-ci-controller` rows holding another key's data | 492 |
| `friday-memory-us-prod` rows holding another key's data | 252 |
| Raw `sk-` / Bearer keys found in these rows | 0 |
| Database / spend-log table | 2,319 MB / 2,285 MB |
| `proxy_server_request`, stored (compressed) | 1,152 MB, of which 283 MB in these rows |
| `previous_models` JSON text | about 500 MB |

### CN (A100)

| | |
|---|---|
| Spend-log rows (2026-11-19 to 11-26) | 175,139 |
| Rows containing `previous_models` | 56,327 |
| Database / spend-log table | 28 GB / 28 GB |
| `proxy_server_request`, stored (compressed) | 24 GB, of which 17 GB in these rows |
| `proxy_server_request` JSON text | 28 GB, of which 19 GB in these rows (largest row 34 MB) |
| ... holding another virtual key's hash (a lower bound for cross-key content) | 361: `friday-memory-cn` 245, `memory-ci-controller` 116 |
| ... of those, also holding an inline image | 99 |
| ... cross-key rows, first and last | 2026-11-24 04:48 to 08:48 UTC |
| Raw `sk-` / Bearer keys found in the rows with `previous_models` | 0 |
| Free space on `/data` | 2.2 TB |

Both regions' cross-key rows were counted the same way (any key hash in the row other than its own).

### Both

Rows with `previous_models` but no other key's hash can still hold other requests of the same key.
The `metadata` and `messages` columns contain none. `previous_response_id` history is rebuilt from
the row's own `input` / `messages`, so other callers' content was not fed back to a model by that path.
Anyone who can read a key's spend logs (admin UI, `/spend/logs`, database dumps) could see it.

## Verification

2026-11-26, on LiteLLM 1.99.0: a real proxy with `config.yaml`'s router, callback and general
settings, a throwaway Postgres, fake order 1 (always 503) and order 2 upstreams, two virtual keys,
chat and Responses requests with inline images.

| Rows holding the other key's text and image | without the hook | with it |
|---|---|---|
| Answered by order 2 (14 rows) | 11 | 0 |
| Every deployment failed (12 rows) | 10 | 0 |

`previous_response_id` carried the earlier turn upstream with the hook (string `input`).

The cleanup SQL ran on 2,814 contaminated rows in three batches: none left, every row equal to its
original minus `previous_models`, rows newer than `fixed_at` untouched, and a cleaned response id
still continued with its history.

## Other copies

Searched 2026-11-26 on both hosts: files newer than 2026-11-19 under `/home`, `/data`, `/srv`,
`/opt`, `/var/backups`, `/mnt`, `/backup` with dump-like names. None readable by `derek` was a copy
of the spend logs. Both databases have WAL archiving off and no replica or replication slot. The
GPU4 proxy logs, the CN `logs/litellm*.log` and the CN Postgres container log never mention
`previous_models`.

Not checked: other users' directories, GPU4's Postgres server log (`adm` group only), files with
other names, and copies off these two hosts.

## Cleanup (not run)

Doing nothing also works: the 7-day retention removes the last affected row 7 days after the fixed
proxy starts. Clean up now if the cross-key content must not stay that long.

Order: restart with the fix first, otherwise new rows keep arriving. This removes only the
breadcrumbs and keeps each request's own body, so `previous_response_id` keeps working. Tested on a
throwaway database (see Verification), not on either live one. A read-only dry run of the `#-`
expression on GPU4 left no `previous_models` in any of the 8,312 rows.

```sql
-- psql -v fixed_at='<UTC time the fixed proxy started>'; newer rows hold the fix's short records.
-- One scan collects the rows; each batch then goes by primary key.
CREATE TEMP TABLE leak_ids AS
SELECT request_id FROM "LiteLLM_SpendLogs"
WHERE "startTime" < :'fixed_at'
  AND (proxy_server_request #> '{metadata,previous_models}' IS NOT NULL
       OR proxy_server_request #> '{litellm_metadata,previous_models}' IS NOT NULL);
WITH batch AS (
  DELETE FROM leak_ids WHERE request_id IN (SELECT request_id FROM leak_ids LIMIT 1000)
  RETURNING request_id)
UPDATE "LiteLLM_SpendLogs" s
SET proxy_server_request = s.proxy_server_request
      #- '{metadata,previous_models}' #- '{litellm_metadata,previous_models}'
FROM batch WHERE s.request_id = batch.request_id;
\watch 2
-- Stop (Ctrl-C) once it reports UPDATE 0.
```

Then `VACUUM (ANALYZE) "LiteLLM_SpendLogs";`. That frees the space for reuse but does not shrink the
files. The update rewrites each affected row, so the table may grow while it runs. `VACUUM FULL`
returns the space but locks out log writes and history lookups while it runs: not worth it on GPU4
(about 280 MB at most), optional on CN in a quiet window (not urgent with 2.2 TB free).

Run it on each region's own database.
