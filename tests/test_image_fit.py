"""Oversized inline images are shrunk before the engine sees them."""
from __future__ import annotations

import base64
import io
import json
import unittest
from unittest import mock

from PIL import Image

import image_fit
import model_manager as mm
from tests.test_model_manager_context_fit import MODEL, RouterHarness

MAX = 10_000
ENGINE_PIXELS = 4_194_304


def url(width, height, fmt="PNG", mode="RGB", **save):
    out = io.BytesIO()
    Image.new(mode, (width, height)).save(out, fmt, **save)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(out.getvalue()).decode()


def opened(data_url):
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def chat(image_url):
    return {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": image_url}}]}]}


class ImageTokenTests(unittest.TestCase):
    """The expected counts are what the engine reported as prompt_tokens, less the text (A100, 2026-11-27)."""

    def test_inline_images_are_counted_by_their_size(self):
        for size, tokens in (((640, 240), 160), ((730, 550), 391), ((900, 500), 448), ((2048, 2048), 4096),
                             ((2364, 1773), 4070), ((10, 10), 64)):
            self.assertEqual(tokens, image_fit.image_tokens(url(*size), ENGINE_PIXELS), size)

    def test_larger_images_count_at_the_cap(self):
        self.assertLessEqual(image_fit.image_tokens(url(4000, 3000), ENGINE_PIXELS), ENGINE_PIXELS // 1024)
        self.assertGreater(image_fit.image_tokens(url(4000, 3000), ENGINE_PIXELS), ENGINE_PIXELS // 1024 - 150)

    def test_an_image_whose_size_is_unknown_counts_as_the_largest(self):
        for value in ("https://example.com/a.png", "data:image/png;base64,AAAA", "data:image/png,AAAA", None):
            self.assertEqual(ENGINE_PIXELS // 1024, image_fit.image_tokens(value, ENGINE_PIXELS), value)


class ImageDetectionTests(unittest.TestCase):
    def test_any_part_that_names_an_image_is_one_whatever_its_type(self):
        """Not left to how an engine reads a part whose declared type does not fit; only an empty field is ignored."""
        def has(part):
            return image_fit.has_images({"messages": [{"role": "user", "content": [part]}]})

        for part in ({"image_url": {"url": "data:,"}}, {"type": "text", "text": "x", "image_url": "file:///x"},
                     {"type": "audio_url", "image_url": {"url": "data:,"}}, {"type": "text", "uuid": "u", "image_url": None}):
            self.assertTrue(has(part), part)
        self.assertFalse(has({"type": "text", "text": "x", "image_url": None}))
        self.assertTrue(image_fit.unsafe_image_url({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "x", "image_url": "file:///x"}]}]}))


class CountedImageTests(unittest.TestCase):
    def test_only_what_the_engine_reads_as_an_image_takes_context(self):
        """A text part is read as text whatever else it carries: billing it an image dropped history needlessly."""
        parts = [{"type": "text", "text": "x", "image_url": "data:,"}, {"type": "image_url", "image_url": {"url": "a"}},
                 {"image_url": "b"}, {"type": "text", "uuid": "u", "image_url": "c"}, {"type": "input_image", "image_url": "d"}]
        self.assertEqual(["a", "b", "c", "d"], list(image_fit.image_urls([{"role": "user", "content": parts}])))


class FitImagesTests(unittest.TestCase):
    def test_large_image_is_shrunk_keeping_shape_and_format(self):
        for fmt in ("PNG", "JPEG"):
            body = chat(url(400, 100, fmt))
            self.assertEqual(1, image_fit.fit_images(body, MAX))
            small = opened(body["messages"][0]["content"][1]["image_url"]["url"])
            self.assertEqual(fmt, small.format)
            self.assertEqual((200, 50), small.size)

    def test_small_remote_and_unreadable_images_are_untouched(self):
        for value in (url(100, 100), "https://example.com/a.png", "data:image/png;base64,bm90IGFuIGltYWdl",
                      "data:image/png;base64,###"):
            body = chat(value)
            self.assertEqual(0, image_fit.fit_images(body, MAX))
            self.assertEqual(value, body["messages"][0]["content"][1]["image_url"]["url"])

    def test_responses_input_and_tool_output_images(self):
        part = {"type": "input_image", "image_url": url(400, 100)}
        body = {"input": [{"role": "user", "content": [part]},
                          {"type": "function_call_output", "output": [dict(part)]}]}
        self.assertEqual(2, image_fit.fit_images(body, MAX))
        self.assertEqual((200, 50), opened(body["input"][1]["output"][0]["image_url"]).size)

    def test_palette_image_and_exif_orientation_survive(self):
        body = chat(url(400, 100, mode="P"))
        self.assertEqual(1, image_fit.fit_images(body, MAX))
        exif = Image.Exif()
        exif[0x0112] = 6   # rotate 90
        body = chat(url(400, 100, "JPEG", exif=exif.tobytes()))
        image_fit.fit_images(body, MAX)
        self.assertEqual(6, opened(body["messages"][0]["content"][1]["image_url"]["url"]).getexif()[0x0112])

    def test_an_oversized_image_in_a_mode_png_cannot_hold_is_still_shrunk(self):
        data = chat(url(300, 200, "TIFF", "CMYK"))
        self.assertEqual(1, image_fit.fit_images(data, MAX))
        small = opened(data["messages"][0]["content"][1]["image_url"]["url"])
        self.assertLessEqual(small.width * small.height, MAX)

    def test_an_oversized_image_that_cannot_be_re_encoded_is_refused(self):
        big = url(300, 200)
        with mock.patch.object(Image.Image, "save", side_effect=OSError("cannot write mode")), \
                self.assertRaises(image_fit.ImageTooLarge):
            image_fit._shrunk_data_url(big, MAX)

    def test_decompression_bomb_is_refused(self):
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1_000), self.assertRaises(image_fit.ImageTooLarge):
            image_fit.fit_images(chat(url(100, 100)), MAX)


class ImageFitInRouterTests(RouterHarness):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        config = self.router.model_configs[MODEL]
        self.router.model_configs[MODEL] = mm.replace(config, max_image_pixels=MAX)

    async def _post(self, payload):
        return await self.client.post("/v1/chat/completions", json=payload)

    async def test_the_engine_receives_the_shrunk_image(self):
        response = await self._post(chat(url(400, 100)))
        self.assertEqual(200, response.status)
        sent = json.loads(self.forwarded[0])
        self.assertEqual((200, 50), opened(sent["messages"][0]["content"][1]["image_url"]["url"]).size)

    async def test_an_undecodably_large_image_is_a_400(self):
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 1_000):
            response = await self._post(chat(url(100, 100)))
        self.assertEqual(400, response.status)
        self.assertEqual([], self.forwarded)

    def test_only_the_27b_is_capped(self):
        self.assertEqual(4_194_304, mm.MODEL_CONFIGS["qwen3.8-27b"].max_image_pixels)
        self.assertIsNone(mm.MODEL_CONFIGS["qwen3.6-35b-a3b-heretic"].max_image_pixels)


if __name__ == "__main__":
    unittest.main()
