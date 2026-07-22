#!/usr/bin/env python3
"""Re-apply the company-domain signup allowlist patch to Open WebUI.

The allowlist that keeps `llm.preseen.ai` from accepting public self-signup is a
LOCAL PATCH inside `open_webui/routers/auths.py` -- a file that lives in
`.venv-owui/` (gitignored, pip-managed).  Any `pip install -U open-webui`
overwrites it and silently reopens public registration.

This script is idempotent: it is a no-op when the patch is already present, and
re-applies it when an upgrade has wiped it.  It is meant to run from
`run_open_webui.sh` before the server starts.

Exit codes -- the caller MUST fail closed on anything non-zero:
    0  patch verified present (either already there, or applied successfully)
    1  patch is absent and could not be applied; signup MUST be disabled
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

# Substring that proves our patch is in the file. Must appear in PATCH_BLOCK.
MARKER = "stardust.AI company-domain allowlist (LOCAL PATCH"

# Insert immediately after this block, inside `async def signup(`. The same two
# lines also appear in `async def add_user(`, so the anchor is only ever
# searched for *after* the signup function's start -- never globally.
SIGNUP_DEF = "async def signup("
ANCHOR = (
    "    if not validate_email_format(form_data.email.lower()):\n"
    "        raise HTTPException(status.HTTP_400_BAD_REQUEST,"
    " detail=ERROR_MESSAGES.INVALID_EMAIL_FORMAT)\n"
)

PATCH_BLOCK = """
    # stardust.AI company-domain allowlist (LOCAL PATCH, not upstream).
    # Restrict email/password signup to allowed company domains. The first
    # user (initial-admin bootstrap) is exempt so the instance can be set up.
    if has_users:
        allowed_domains = [
            domain.strip().lower()
            for domain in os.environ.get('SIGNUP_ALLOWED_EMAIL_DOMAINS', 'stardust.ai').split(',')
            if domain.strip()
        ]
        email_domain = form_data.email.lower().rsplit('@', 1)[-1]
        if allowed_domains and email_domain not in allowed_domains:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail='Registration is restricted to stardust.AI company accounts.',
            )
"""

GLOB = ".venv-owui/lib/python*/site-packages/open_webui/routers/auths.py"


def _default_target() -> Path:
    """Locate auths.py inside the sibling .venv-owui, whatever the python version."""
    root = Path(__file__).resolve().parent.parent
    matches = sorted(root.glob(GLOB))
    if matches:
        return matches[-1]
    # Fall back to a conventional path so the error message names something real.
    return root / ".venv-owui/lib/python3.12/site-packages/open_webui/routers/auths.py"


def _patch_is_inside_signup(source: str) -> bool:
    """True only when MARKER sits within the body of `async def signup(`.

    Guards against the patch landing in `add_user` (which shares the anchor) or
    surviving somewhere harmless while signup itself is left unguarded.
    """
    signup_at = source.find(SIGNUP_DEF)
    marker_at = source.find(MARKER)
    if signup_at == -1 or marker_at == -1 or marker_at < signup_at:
        return False
    # The marker must appear before the next top-level def that follows signup.
    next_def = source.find("\nasync def ", signup_at + len(SIGNUP_DEF))
    return next_def == -1 or marker_at < next_def


def _apply(source: str) -> str:
    signup_at = source.find(SIGNUP_DEF)
    if signup_at == -1:
        raise RuntimeError(f"cannot locate `{SIGNUP_DEF}` -- upstream renamed the signup route")

    anchor_at = source.find(ANCHOR, signup_at)
    if anchor_at == -1:
        raise RuntimeError(
            "cannot locate the email-validation anchor inside signup() -- "
            "upstream changed the function body; patch must be re-derived by hand"
        )

    next_def = source.find("\nasync def ", signup_at + len(SIGNUP_DEF))
    if next_def != -1 and anchor_at > next_def:
        raise RuntimeError("anchor found outside signup() -- refusing to patch the wrong function")

    cut = anchor_at + len(ANCHOR)
    return source[:cut] + PATCH_BLOCK + source[cut:]


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_target()

    if not target.is_file():
        print(f"[owui-patch] FAIL: {target} does not exist", file=sys.stderr)
        return 1

    try:
        original = target.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[owui-patch] FAIL: cannot read {target}: {exc}", file=sys.stderr)
        return 1

    if _patch_is_inside_signup(original):
        print("[owui-patch] OK: domain allowlist already present in signup()")
        return 0

    print("[owui-patch] patch MISSING (likely an open-webui upgrade) -- re-applying", file=sys.stderr)

    try:
        patched = _apply(original)
    except RuntimeError as exc:
        print(f"[owui-patch] FAIL: {exc}", file=sys.stderr)
        return 1

    # A syntactically broken auths.py would take the whole service down, which is
    # a worse outcome than an unpatched one. Validate before writing anything.
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        print(f"[owui-patch] FAIL: patched file would not parse: {exc}", file=sys.stderr)
        return 1

    backup = target.with_suffix(target.suffix + ".prepatch")
    try:
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(patched, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        print(f"[owui-patch] FAIL: cannot write {target}: {exc}", file=sys.stderr)
        return 1

    # Re-read from disk: this verification result -- not the write succeeding --
    # is what the caller's fail-closed branch depends on.
    try:
        verified = target.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[owui-patch] FAIL: cannot re-read {target}: {exc}", file=sys.stderr)
        return 1

    if not _patch_is_inside_signup(verified):
        print("[owui-patch] FAIL: patch not detectable after write", file=sys.stderr)
        return 1

    print("[owui-patch] OK: domain allowlist re-applied to signup()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
