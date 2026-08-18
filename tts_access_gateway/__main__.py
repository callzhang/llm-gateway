"""Process entrypoint for the TTS access gateway."""

from __future__ import annotations

import logging

from aiohttp import web

from .app import create_app
from .config import GatewayConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = GatewayConfig.from_env()
    web.run_app(
        create_app(config),
        host=config.listen_host,
        port=config.listen_port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
