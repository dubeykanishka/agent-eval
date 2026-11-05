import hashlib
import json
import logging
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.budget import Budget, BudgetExceeded, SpendTracker
from src.config import ModelSpec

log = logging.getLogger(__name__)

RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,   # includes timeouts
    anthropic.InternalServerError,
)


class LLM:
    def __init__(self, spec: ModelSpec, cache_dir: str | Path = "cache",
                 client=None, tracker: SpendTracker | None = None):
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        if client is None:
            load_dotenv()
            client = anthropic.Anthropic(api_key=os.environ["PROVIDER_API_KEY"])
        self.client = client

    # ---------- helpers ----------
    def _key(self, system, messages, tools, max_tokens, temperature) -> str:
        blob = json.dumps(
            [self.spec.model_id, system, messages, tools, max_tokens, temperature],
            sort_keys=True, default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _estimate_tokens(system, messages, tools) -> int:
        # Deliberately pessimistic (3 chars per token). Real counts come from the API.
        chars = len(json.dumps([system, messages, tools], default=str))
        return chars // 3 + 1

    # ---------- the one place that touches the network ----------
    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    def _raw_call(self, system, messages, tools, max_tokens, temperature) -> dict:
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=max_tokens,
            messages=messages,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)

        content = [b.model_dump(exclude_none=True) for b in resp.content]
        text = "".join(b.get("text", "") for b in content if b["type"] == "text")
        tool_calls = [
            {"id": b["id"], "name": b["name"], "args": b["input"]}
            for b in content if b["type"] == "tool_use"
        ]
        u = resp.usage
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        return dict(
            text=text,
            tool_calls=tool_calls,
            content=content,                    # raw blocks, to append as the assistant turn
            stop_reason=resp.stop_reason,
            tin=u.input_tokens + cache_read + cache_write,   # raw tokens, as defined in section 11
            tout=u.output_tokens,
        )

    # ---------- public API ----------
    def complete(self, messages, budget: Budget, tag: str, system=None, tools=None,
                 max_tokens: int = 1024, temperature: float | None = None) -> dict:
        if self.tracker:
            self.tracker.check()

        # Pre-flight: refuse calls that cannot fit, and cap output to what is left.
        est_in = self._estimate_tokens(system, messages, tools)
        room = budget.remaining() - est_in
        if room < 64:
            raise BudgetExceeded(
                f"{tag}: ~{est_in} input tokens do not fit in {budget.remaining()} remaining"
            )
        max_tokens = min(max_tokens, room)

        path = self.cache_dir / f"{self._key(system, messages, tools, max_tokens, temperature)}.json"
        cached = path.exists()
        if cached:
            r = json.loads(path.read_text())
            latency = 0.0
        else:
            t0 = time.time()
            r = self._raw_call(system, messages, tools, max_tokens, temperature)
            latency = time.time() - t0
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(r))
            os.replace(tmp, path)               # atomic: safe with threads

        usd = self.spec.cost_usd(r["tin"], r["tout"])
        thr = self.spec.long_context_threshold
        if thr and r["tin"] > thr:
            log.warning("%s: prompt of %d tokens crossed the %d long-context line", tag, r["tin"], thr)

        # A cache hit still charges the EXPERIMENT budget (it should behave as if paid),
        # but costs nothing real, so the spend tracker ignores it.
        budget.charge(r["tin"], r["tout"], usd, latency, tag, cached=cached)
        if self.tracker and not cached:
            self.tracker.add(usd)
        return {**r, "usd": usd, "cached": cached}