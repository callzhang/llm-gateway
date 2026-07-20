#!/usr/bin/env python3.12
from open_webui.routers.chats import _extract_svg_artifact_from_message


def test_extracts_svg_from_xml_fenced_block():
    message = {
        "id": "assistant-message",
        "role": "assistant",
        "content": """
Here is the file:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
  <path d="M0 0h10v10H0z"/>
</svg>
```
""",
    }

    artifact = _extract_svg_artifact_from_message(message)

    assert artifact is not None
    assert artifact["filename"] == "assistant-message.svg"
    assert artifact["content"].startswith("<svg")
    assert artifact["content"].endswith("</svg>")


def test_ignores_existing_svg_attachment():
    message = {
        "id": "assistant-message",
        "role": "assistant",
        "files": [{"name": "already.svg", "content_type": "image/svg+xml"}],
        "content": "```svg\n<svg></svg>\n```",
    }

    assert _extract_svg_artifact_from_message(message) is None


def test_extracts_svg_embedded_in_python_fenced_block():
    message = {
        "id": "assistant-message",
        "role": "assistant",
        "content": '''
```python
svg_content = """
<svg xmlns="http://www.w3.org/2000/svg">
  <circle cx="5" cy="5" r="5"/>
</svg>
"""
```
''',
    }

    artifact = _extract_svg_artifact_from_message(message)

    assert artifact is not None
    assert "<circle" in artifact["content"]


if __name__ == "__main__":
    test_extracts_svg_from_xml_fenced_block()
    test_ignores_existing_svg_attachment()
    test_extracts_svg_embedded_in_python_fenced_block()
