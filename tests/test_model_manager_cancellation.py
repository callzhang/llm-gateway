from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from aiohttp import web

import model_manager as mm


class SharedStartupCancellationTests(unittest.IsolatedAsyncioTestCase):
    def backend(self):
        backend = mm.GpuBackend("qwen3.8-27b", "unused.sh", "qwen3.8-27b", mm.GpuSlot(0, 0, 9000))
        backend.slot.backend = backend
        return backend

    async def test_cancelled_waiter_does_not_restart_shared_startup(self):
        backend = self.backend()
        entered, release = asyncio.Event(), asyncio.Event()
        starts = 0

        async def spawn():
            nonlocal starts
            starts += 1
            entered.set()
            await release.wait()
            backend.process = SimpleNamespace(poll=lambda: None)
            backend._ready = True

        with patch.object(backend, "_spawn_locked", side_effect=spawn):
            first = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(backend._ensure_running())
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            await asyncio.wait_for(second, 1)
            self.assertEqual(1, starts)
            self.assertTrue(backend._ready)

    async def test_startup_failure_releases_slot_after_only_waiter_disconnects(self):
        backend = self.backend()
        entered, release = asyncio.Event(), asyncio.Event()

        async def spawn():
            entered.set()
            await release.wait()
            raise RuntimeError("controlled startup failure")

        with patch.object(backend, "_spawn_locked", side_effect=spawn):
            waiter = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            release.set()
            # Yield to the backend-owned startup and its completion callback.
            for _ in range(10):
                await asyncio.sleep(0)
            self.assertTrue(backend._failed)
            self.assertIsNone(backend.slot.backend)

    async def test_explicit_stop_cancels_startup_and_releases_slot(self):
        backend = self.backend()
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def spawn():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(backend, "_spawn_locked", side_effect=spawn), patch.object(backend, "_kill_process_locked"):
            waiter = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(backend.stop(), 1)
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertTrue(cancelled.is_set())
            self.assertIsNone(backend.slot.backend)

    async def test_failure_reaches_all_waiters_without_restarting(self):
        backend = self.backend()
        entered, release = asyncio.Event(), asyncio.Event()
        failure = RuntimeError("controlled startup failure")

        async def spawn():
            entered.set()
            await release.wait()
            raise failure

        with patch.object(backend, "_spawn_locked", side_effect=spawn) as spawn_mock:
            first = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            second = asyncio.create_task(backend._ensure_running())
            release.set()
            results = await asyncio.gather(first, second, return_exceptions=True)
            self.assertIs(failure, results[0])
            self.assertIs(failure, results[1])
            self.assertEqual(1, spawn_mock.call_count)
            self.assertTrue(backend._failed)
            self.assertIsNone(backend.slot.backend)

    async def test_session_shutdown_reaps_unready_spawn_but_preserves_ready_model(self):
        backend = self.backend()
        entered = asyncio.Event()
        process = SimpleNamespace(pid=654321, poll=lambda: None, wait=lambda: 0)

        async def spawn():
            backend.process = process
            entered.set()
            await asyncio.Event().wait()

        with patch.object(backend, "_spawn_locked", side_effect=spawn), patch.object(mm.os, "killpg") as kill:
            waiter = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            await backend._close_session()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            kill.assert_called_once_with(process.pid, mm.signal.SIGKILL)
            self.assertIsNone(backend.process)
            self.assertIsNone(backend.slot.backend)
        ready = self.backend()
        ready.process, ready._ready = process, True
        with patch.object(mm.os, "killpg") as kill:
            await ready._close_session()
            kill.assert_not_called()
            self.assertIs(process, ready.process)

    async def test_orphaned_startup_finishes_without_an_unretrieved_exception(self):
        backend = self.backend()
        entered, release = asyncio.Event(), asyncio.Event()
        errors = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: errors.append(context))

        async def spawn():
            entered.set()
            await release.wait()
            raise RuntimeError("controlled orphan failure")

        try:
            with patch.object(backend, "_spawn_locked", side_effect=spawn):
                waiter = asyncio.create_task(backend._ensure_running())
                await asyncio.wait_for(entered.wait(), 1)
                waiter.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
                release.set()
                for _ in range(10):
                    await asyncio.sleep(0)
                self.assertTrue(backend._startup_task.done())
                self.assertFalse(backend._startup_task._log_traceback)
                self.assertEqual([], errors)
        finally:
            loop.set_exception_handler(previous)

    async def test_shutdown_blocks_new_startup_while_reaping_old_process(self):
        backend = self.backend()
        entered, reaping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        starts = 0
        process = SimpleNamespace(pid=654321, poll=lambda: None, wait=lambda: 0)

        async def spawn():
            nonlocal starts
            starts += 1
            backend.process = process
            if starts == 1:
                entered.set()
                await asyncio.Event().wait()
            else:
                backend._ready = True

        async def reap(_wait):
            reaping.set()
            await release.wait()

        with patch.object(backend, "_spawn_locked", side_effect=spawn), patch.object(mm.os, "killpg"), patch.object(mm.asyncio, "to_thread", side_effect=reap):
            waiter = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            cleanup = asyncio.create_task(backend._close_session())
            await asyncio.wait_for(reaping.wait(), 1)
            try:
                self.assertFalse(backend.slot.is_free)
                router = mm.DynamicRouter([backend.slot], {"qwen3.8-27b": mm.ModelConfig(script="unused.sh", served_name="qwen3.8-27b", max_num_seqs=8)})
                self.assertEqual([], router._free_slots())
                with self.assertRaises(RuntimeError):
                    await backend._ensure_running()
                self.assertEqual(1, starts)
            finally:
                release.set()
                await asyncio.wait_for(cleanup, 1)
                with self.assertRaises(asyncio.CancelledError):
                    await waiter

    async def test_cancelled_shutdown_caller_does_not_abandon_process_reap(self):
        backend = self.backend()
        entered, reaping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        process = SimpleNamespace(pid=654321, poll=lambda: None, wait=lambda: 0)

        async def spawn():
            backend.process = process
            entered.set()
            await asyncio.Event().wait()

        async def reap(_wait):
            reaping.set()
            await release.wait()

        with patch.object(backend, "_spawn_locked", side_effect=spawn), patch.object(mm.os, "killpg"), patch.object(mm.asyncio, "to_thread", side_effect=reap):
            waiter = asyncio.create_task(backend._ensure_running())
            await asyncio.wait_for(entered.wait(), 1)
            cleanup = asyncio.create_task(backend._close_session())
            await asyncio.wait_for(reaping.wait(), 1)
            cleanup.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cleanup
            try:
                self.assertFalse(backend._startup_shutdown_task.cancelled())
            finally:
                release.set()
                await asyncio.gather(backend._startup_shutdown_task, return_exceptions=True)
                with self.assertRaises(asyncio.CancelledError):
                    await waiter
            self.assertIsNone(backend.process)
            self.assertIsNone(backend.slot.backend)


class HttpCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []
        self.clients = []
        self.backends = []
        self.upstream_entered = asyncio.Event()
        self.upstream_cancelled = asyncio.Event()
        self.upstream_release = asyncio.Event()
        self.proxy_done = asyncio.Event()

    async def asyncTearDown(self):
        self.upstream_release.set()
        for client in self.clients:
            await client.close()
        for runner in reversed(self.runners):
            await runner.cleanup()
        for backend in self.backends:
            await backend._close_session()

    async def serve(self, app, *, production=False):
        runner = web.AppRunner(app) if production else web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.runners.append(runner)
        return site._server.sockets[0].getsockname()[1]

    async def start_proxy(self, *, streaming=False):
        async def upstream(request):
            await request.read()
            response = web.StreamResponse()
            if streaming:
                await response.prepare(request)
                await response.write(b"first chunk\n")
            self.upstream_entered.set()
            try:
                await self.upstream_release.wait()
            except asyncio.CancelledError:
                self.upstream_cancelled.set()
                raise
            return response if streaming else web.Response(body=b"complete")

        app = web.Application()
        app.router.add_post("/v1/chat/completions", upstream)
        upstream_port = await self.serve(app)
        backend = mm.GpuBackend("qwen3.8-27b", "unused.sh", "qwen3.8-27b", mm.GpuSlot(0, 0, upstream_port))
        backend._ensure_session()
        self.backends.append(backend)

        async def proxy(request):
            try:
                return await backend.proxy(request, await request.read())
            finally:
                self.proxy_done.set()

        app = web.Application()
        app.router.add_post("/v1/chat/completions", proxy)
        proxy_port = await self.serve(app, production=True)
        client = aiohttp.ClientSession()
        self.clients.append(client)
        return backend, client, f"http://127.0.0.1:{proxy_port}/v1/chat/completions"

    async def test_disconnect_before_headers_cancels_upstream_and_releases_counter(self):
        backend, client, url = await self.start_proxy()
        pending = asyncio.create_task(client.post(url, data=b"{}"))
        await asyncio.wait_for(self.upstream_entered.wait(), 1)
        self.assertEqual(1, backend._active_requests)
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        await client.close()
        await asyncio.wait_for(self.proxy_done.wait(), 1)
        await asyncio.wait_for(self.upstream_cancelled.wait(), 1)
        self.assertEqual(0, backend._active_requests)

    @asynccontextmanager
    async def configured_responses_proxy(self, client, url):
        # Characterize our configured LiteLLM boundary, not a mock cancellation:
        # the real aresponses processor owns a real HTTP call to GpuBackend.proxy.
        # Only model routing/auth setup are controlled; no model is started.
        from pathlib import Path

        import yaml
        from fastapi import Response
        from starlette.requests import Request
        from litellm.proxy import common_request_processing as processing
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.llms.openai import ResponsesAPIResponse

        async def infer():
            async with client.post(url, data=b"{}") as response:
                self.assertEqual(200, response.status)
                body = (await response.read()).decode()
            # Controlled model adapter only; the real HTTP body must arrive before
            # constructing a real typed Responses result for the processor tail.
            return ResponsesAPIResponse(
                id="resp-synthetic", created_at=0, model="synthetic-cancellation",
                object="response", status="completed",
                output=[{"type": "message", "role": "assistant", "id": "msg-synthetic",
                         "status": "completed", "content": [
                             {"type": "output_text", "text": body, "annotations": []}]}],
            )

        async def route(*, data, route_type, **_unused):
            if route_type != "aresponses" or data["model"] != "synthetic-cancellation":
                raise AssertionError("unexpected route at the controlled model boundary")
            return infer()

        async def no_op(**_unused):
            return None

        async def successful_response(*, response, **_unused):
            return response

        # Do not send real config credentials into the synthetic request.
        settings = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
        general_settings = {
            "cancel_on_disconnect": settings.get("general_settings", {}).get("cancel_on_disconnect", False)
        }
        logging = SimpleNamespace(
            during_call_hook=no_op, update_request_status=no_op,
            post_call_success_hook=successful_response, post_call_response_headers_hook=no_op,
        )
        pending_requests = []

        def start_request():
            received, monitor_closed = asyncio.Queue(), asyncio.Event()

            async def receive():
                try:
                    return await received.get()
                finally:
                    monitor_closed.set()

            request = Request(
                {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []},
                receive=receive,
            )
            processor = processing.ProxyBaseLLMRequestProcessing({
                "model": "synthetic-cancellation", "stream": False,
                "litellm_logging_obj": SimpleNamespace(
                    litellm_call_id="synthetic-call", litellm_params={},
                    model_call_details={"response_cost": 0},
                ),
            })
            pending = asyncio.create_task(processor.base_process_llm_request(
                request=request, fastapi_response=Response(),
                user_api_key_dict=UserAPIKeyAuth(), route_type="aresponses",
                proxy_logging_obj=logging,
                general_settings=general_settings, proxy_config=SimpleNamespace(),
                skip_pre_call_logic=True,
            ))
            pending_requests.append(pending)
            return pending, received, monitor_closed

        with patch.object(processing, "route_request", side_effect=route):
            try:
                yield start_request
            finally:
                for pending in pending_requests:
                    if not pending.done():
                        pending.cancel()
                await asyncio.gather(*pending_requests, return_exceptions=True)

    async def test_configured_responses_proxy_disconnect_reaches_gateway_upstream(self):
        from fastapi import HTTPException

        backend, client, url = await self.start_proxy()
        async with self.configured_responses_proxy(client, url) as start_request:
            pending, received, _ = start_request()
            try:
                await asyncio.wait_for(self.upstream_entered.wait(), 3)
                self.assertEqual(1, backend._active_requests)
                received.put_nowait({"type": "http.disconnect"})
                completed, _ = await asyncio.wait({pending}, timeout=1)
                self.assertTrue(completed, "configured Responses proxy retained the disconnected request")
                with self.assertRaises(HTTPException) as cancelled:
                    await pending
                self.assertEqual(499, cancelled.exception.status_code)
                await asyncio.wait_for(self.proxy_done.wait(), 1)
                await asyncio.wait_for(self.upstream_cancelled.wait(), 1)
                self.assertEqual(0, backend._active_requests)
            finally:
                pending.cancel()
                with suppress(asyncio.CancelledError, HTTPException):
                    await pending

    async def test_configured_responses_proxy_normal_request_completes(self):
        backend, client, url = await self.start_proxy()
        async with self.configured_responses_proxy(client, url) as start_request:
            pending, _, monitor_closed = start_request()
            await asyncio.wait_for(self.upstream_entered.wait(), 3)
            self.upstream_release.set()
            response = await asyncio.wait_for(pending, 3)
            self.assertEqual("complete", response.output_text)
            self.assertEqual("completed", response.status)
            self.assertEqual("synthetic-cancellation", response.model)
            self.assertFalse(self.upstream_cancelled.is_set())
            await asyncio.wait_for(monitor_closed.wait(), 1)
            self.assertEqual(0, backend._active_requests)

    async def test_configured_responses_proxy_disconnect_does_not_cancel_peer(self):
        from fastapi import HTTPException

        backend, client, url = await self.start_proxy()
        async with self.configured_responses_proxy(client, url) as start_request:
            first, received, _ = start_request()
            await asyncio.wait_for(self.upstream_entered.wait(), 3)
            peer, _, peer_monitor_closed = start_request()
            async with asyncio.timeout(3):
                while backend._active_requests != 2:
                    await asyncio.sleep(0.001)
            received.put_nowait({"type": "http.disconnect"})
            with self.assertRaises(HTTPException) as cancelled:
                await asyncio.wait_for(first, 1)
            self.assertEqual(499, cancelled.exception.status_code)
            await asyncio.wait_for(self.proxy_done.wait(), 1)
            await asyncio.wait_for(self.upstream_cancelled.wait(), 1)
            self.assertEqual(1, backend._active_requests)
            self.assertFalse(peer.done())
            self.upstream_release.set()
            response = await asyncio.wait_for(peer, 3)
            self.assertEqual("complete", response.output_text)
            self.assertEqual("completed", response.status)
            await asyncio.wait_for(peer_monitor_closed.wait(), 1)
            self.assertEqual(0, backend._active_requests)

    async def test_disconnect_during_stream_cancels_upstream_and_releases_counter(self):
        backend, client, url = await self.start_proxy(streaming=True)
        response = await client.post(url, data=b"{}")
        self.assertEqual(b"first chunk\n", await response.content.readline())
        response.close()
        await client.close()
        await asyncio.wait_for(self.proxy_done.wait(), 1)
        await asyncio.wait_for(self.upstream_cancelled.wait(), 1)
        self.assertEqual(0, backend._active_requests)

    async def test_normal_response_preserves_body_status_and_counter(self):
        backend, client, url = await self.start_proxy()
        self.upstream_release.set()
        async with client.post(url, data=b"{}") as response:
            self.assertEqual(200, response.status)
            self.assertEqual(b"complete", await response.read())
        await asyncio.wait_for(self.proxy_done.wait(), 1)
        self.assertFalse(self.upstream_cancelled.is_set())
        self.assertEqual(0, backend._active_requests)

    async def test_cancelling_one_request_does_not_cancel_another_request(self):
        backend, client, url = await self.start_proxy()
        first = asyncio.create_task(client.post(url, data=b"{}"))
        await asyncio.wait_for(self.upstream_entered.wait(), 1)
        other_client = aiohttp.ClientSession()
        self.clients.append(other_client)
        second = asyncio.create_task(other_client.post(url, data=b"{}"))
        for _ in range(100):
            if backend._active_requests == 2:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(2, backend._active_requests)
        first.cancel()
        with suppress(asyncio.CancelledError):
            await first
        await client.close()
        await asyncio.wait_for(self.proxy_done.wait(), 1)
        self.assertEqual(1, backend._active_requests)
        self.assertFalse(second.done())
        self.upstream_release.set()
        response = await asyncio.wait_for(second, 1)
        self.assertEqual(200, response.status)
        self.assertEqual(b"complete", await response.read())
        response.close()
        for _ in range(100):
            if backend._active_requests == 0:
                break
            await asyncio.sleep(0.001)
        self.assertEqual(0, backend._active_requests)

    async def test_non_proxy_handler_keeps_its_shared_protocol_lifetime(self):
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        cancelled = []

        async def shared_protocol(request):
            await request.read()
            entered.set()
            try:
                await release.wait()
                finished.set()
                return web.Response(body=b"protocol drained")
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        app = web.Application()
        app.router.add_post("/v1/audio/transcriptions", shared_protocol)
        port = await self.serve(app, production=True)
        client = aiohttp.ClientSession()
        self.clients.append(client)
        request = asyncio.create_task(client.post(f"http://127.0.0.1:{port}/v1/audio/transcriptions", data=b"audio"))
        await asyncio.wait_for(entered.wait(), 1)
        request.cancel()
        with suppress(asyncio.CancelledError):
            await request
        await client.close()
        # Let connection-close handling run, then verify protocol drain is still alive.
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual([], cancelled)
