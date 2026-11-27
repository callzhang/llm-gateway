"""SGLang plugin: keep the API keys out of the engine log.

SGLang 0.5.17 logs its whole ServerArgs at startup, the key it read from --config included.
SGLang calls `register` before building ServerArgs, in every process it starts (entry point
in friday_redact_keys-1.0.dist-info, enabled by SGLANG_PLUGINS in the run script).
"""


def register() -> None:
    from sglang.srt.server_args import ServerArgs

    original = ServerArgs.__repr__

    def redacted(self) -> str:
        text = original(self)
        for name in ("api_key", "admin_api_key"):
            value = getattr(self, name, None)
            if value:
                text = text.replace(f"{name}={value!r}", f"{name}='***'")
        return text

    ServerArgs.__repr__ = redacted
