#!/usr/bin/env python3.12
import asyncio
import sys

from open_webui.internal.db import get_async_db_context
from open_webui.models.chats import Chats
from open_webui.models.users import Users
from open_webui.routers.chats import _attach_svg_artifacts_to_chat_messages


async def main(chat_id: str) -> int:
    async with get_async_db_context() as db:
        chat = await Chats.get_chat_by_id(chat_id, db=db)
        if not chat:
            print(f"chat not found: {chat_id}", file=sys.stderr)
            return 1

        user = await Users.get_user_by_id(chat.user_id, db=db)
        if not user:
            print(f"user not found: {chat.user_id}", file=sys.stderr)
            return 1

        updated_chat = chat.chat
        before = sum(
            len((message or {}).get("files") or [])
            for message in (updated_chat.get("history") or {}).get("messages", {}).values()
            if isinstance(message, dict)
        )

        await _attach_svg_artifacts_to_chat_messages(chat_id, updated_chat, user, db)
        await Chats.update_chat_by_id(chat_id, updated_chat, db=db)
        await Chats.reconcile_messages_by_chat_id(
            chat_id,
            user.id,
            (updated_chat.get("history") or {}).get("messages") or {},
        )

        after = sum(
            len((message or {}).get("files") or [])
            for message in (updated_chat.get("history") or {}).get("messages", {}).values()
            if isinstance(message, dict)
        )
        print(f"attached_files_delta={after - before}")
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: backfill_open_webui_svg_artifacts.py CHAT_ID", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
