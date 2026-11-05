import threading
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """A single run has no token budget left for this call."""


class SpendLimitExceeded(Exception):
    """The whole experiment hit its dollar kill switch."""


@dataclass
class Budget:
    """Token budget for ONE run. Every call a system makes is charged here."""
    limit_tokens: int
    used_in: int = 0
    used_out: int = 0
    calls: list = field(default_factory=list)

    @property
    def used(self) -> int:
        return self.used_in + self.used_out

    def remaining(self) -> int:
        return self.limit_tokens - self.used

    @property
    def overshoot(self) -> int:
        return max(0, self.used - self.limit_tokens)

    def charge(self, tin: int, tout: int, usd: float, latency: float,
               tag: str, cached: bool = False) -> None:
        self.used_in += tin
        self.used_out += tout
        self.calls.append(dict(tag=tag, tin=tin, tout=tout, usd=usd,
                               latency=latency, cached=cached))

    @property
    def usd(self) -> float:
        return sum(c["usd"] for c in self.calls)


class SpendTracker:
    """Shared across all runs and threads. Counts only REAL spend (cache misses)."""

    def __init__(self, limit_usd: float):
        self.limit_usd = limit_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def check(self) -> None:
        if self.spent >= self.limit_usd:
            raise SpendLimitExceeded(f"spent ${self.spent:.2f} of ${self.limit_usd:.2f}")

    def add(self, usd: float) -> None:
        with self._lock:
            self.spent += usd