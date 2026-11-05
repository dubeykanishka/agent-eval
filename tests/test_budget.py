import pytest

from src.budget import Budget, BudgetExceeded, SpendLimitExceeded, SpendTracker
from src.config import load_models
from src.llm import LLM


class FakeLLM(LLM):
    def __init__(self, tmp_path, tracker=None):
        super().__init__(load_models()["worker"], cache_dir=tmp_path,
                         client=object(), tracker=tracker)
        self.n_calls = 0

    def _raw_call(self, system, messages, tools, max_tokens, temperature):
        self.n_calls += 1
        return dict(text="hi", tool_calls=[], content=[{"type": "text", "text": "hi"}],
                    stop_reason="end_turn", tin=1000, tout=50)


MSG = [{"role": "user", "content": "hello"}]


def test_charge_adds_up(tmp_path):
    llm, b = FakeLLM(tmp_path), Budget(limit_tokens=10_000)
    llm.complete(MSG, b, tag="a")
    assert b.used_in == 1000 and b.used_out == 50
    assert b.remaining() == 10_000 - 1050


def test_cache_hit_charges_budget_but_not_wallet(tmp_path):
    tracker = SpendTracker(limit_usd=1.0)
    llm, b = FakeLLM(tmp_path, tracker), Budget(limit_tokens=10_000)
    r1 = llm.complete(MSG, b, tag="a")
    r2 = llm.complete(MSG, b, tag="a")
    assert llm.n_calls == 1                       # second call came from disk
    assert r2["cached"] and not r1["cached"]
    assert b.used == 2 * 1050                     # budget charged both times
    assert tracker.spent == pytest.approx(0.000125)   # wallet charged once


def test_budget_exceeded_when_no_room(tmp_path):
    llm, b = FakeLLM(tmp_path), Budget(limit_tokens=20)
    with pytest.raises(BudgetExceeded):
        llm.complete(MSG, b, tag="a")


def test_spend_limit_kill_switch(tmp_path):
    tracker = SpendTracker(limit_usd=0.0001)      # smaller than one call's cost
    llm = FakeLLM(tmp_path, tracker)
    llm.complete(MSG, Budget(limit_tokens=10_000), tag="a")
    with pytest.raises(SpendLimitExceeded):
        llm.complete([{"role": "user", "content": "different"}],
                     Budget(limit_tokens=10_000), tag="b")