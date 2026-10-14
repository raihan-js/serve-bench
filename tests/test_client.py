"""Client unit tests (no server, no GPU): stats math + a fake SSE server."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiohttp import web

from bench.client import one_request, percentile, run_fixed, summarize


def test_percentile():
    assert percentile([], 0.5) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_summarize():
    recs = [{"status": 200, "start_t": 0.0, "end_t": 1.0, "ttft_s": 0.1,
             "tpot_s": 0.02},
            {"status": 200, "start_t": 0.0, "end_t": 2.0, "ttft_s": 0.3,
             "tpot_s": 0.04},
            {"status": 500, "start_t": 0.0, "end_t": 2.0}]
    s = summarize(recs)
    assert s == {"n": 3, "ok": 2, "err_rate": 0.3333, "ttft_p50": 0.3,
                 "ttft_p95": 0.3, "tpot_p50": 0.04, "tpot_p95": 0.04,
                 "req_per_s": 1.0}


async def _fake_app():
    async def completions(request: web.Request) -> web.Response:
        resp = web.StreamResponse()
        await resp.prepare(request)
        for tok in ["data: a\n\n", "data: b\n\n", "data: [DONE]\n\n"]:
            await resp.write(tok.encode())
            await asyncio.sleep(0.01)
        return resp
    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    return app


def test_run_fixed_against_fake():
    async def go() -> None:
        from aiohttp.test_utils import TestClient, TestServer

        client = TestClient(TestServer(await _fake_app()))
        await client.start_server()
        try:
            import aiohttp

            async with aiohttp.ClientSession() as s:
                r = await one_request(s, str(client.make_url("/v1/completions")),
                                      "m", "hello", max_tokens=8)
            assert r["status"] == 200
            assert r["ttft_s"] >= 0 and r["tpot_s"] >= 0
            recs = await run_fixed(str(client.make_url("/v1/completions")),
                                   "m", ["a", "b", "c"], concurrency=2)
            assert len(recs) == 3
            assert summarize(recs)["ok"] == 3
        finally:
            await client.close()
    asyncio.run(go())
