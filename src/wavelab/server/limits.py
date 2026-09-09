"""Usage limits. Mandatory before putting any of this on the internet.

Every validation costs between 5 and 10 seconds of CPU: the 250 random control filters are not
free, and they are precisely what makes the verdict worth anything. Without limits, anyone with a
three-line loop leaves the service unusable for everybody else — and the node shares a machine with
applications that bring in money.

Three layers, each for a different problem:

1. **Global concurrency.** At most N validations at a time. Protects the node's CPU.
2. **Per-IP bucket.** One user cannot hog the service even when the node is idle.
3. **Body size.** A 500 MB CSV is not a user, it is an attack.

Deliberately, the number of random controls is NOT reduced under load: that would be degrading the
honesty of the result in silence, and a badly computed verdict is worse than one that takes a while
or one that is refused with a clear message.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

__all__ = ["RateLimiter", "TooBusy", "TooMany"]


class TooBusy(RuntimeError):
    """Too many validations running at once."""


class TooMany(RuntimeError):
    """This IP has spent its quota."""


@dataclass
class RateLimiter:
    max_concurrent: int = 2
    per_hour: int = 30
    per_minute: int = 6
    #: How long a caller queues for a free slot before being told to come back. The default is the
    #: production value and is part of the contract; it is a field only so that the test for the
    #: give-up path does not have to spend 25 real seconds of every CI run waiting for it.
    acquire_timeout_s: float = 25.0
    _sem: asyncio.Semaphore | None = None
    _hist: dict[str, deque] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent)

    def _check_ip(self, ip: str) -> None:
        now = time.monotonic()
        q = self._hist.setdefault(ip, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            wait = int(3600 - (now - q[0]))
            raise TooMany(
                f"you have run {self.per_hour} validations in the last hour, which is the limit. "
                f"Come back in {wait // 60} min. Every validation runs 250 random controls and "
                "that costs real CPU.")
        recent = sum(1 for t in q if now - t < 60)
        if recent >= self.per_minute:
            raise TooMany(
                f"at most {self.per_minute} validations per minute. Wait a few seconds.")
        q.append(now)

        # Pruning of idle IPs: without this the dictionary grows without bound and it is a slow
        # memory leak, the kind you only notice after weeks in production.
        if len(self._hist) > 5000:
            for k in [k for k, v in self._hist.items() if not v or now - v[-1] > 7200]:
                self._hist.pop(k, None)

    class _Ctx:
        def __init__(self, lim: RateLimiter, ip: str) -> None:
            self.lim, self.ip = lim, ip

        async def __aenter__(self):
            self.lim._check_ip(self.ip)
            try:
                await asyncio.wait_for(self.lim._sem.acquire(),
                                       timeout=self.lim.acquire_timeout_s)
            except TimeoutError:
                raise TooBusy(
                    "there are too many validations under way right now. Try again in a minute: "
                    "each one takes a few seconds and only two run at a time so that nobody's "
                    "result gets degraded.") from None
            return self

        async def __aexit__(self, *exc):
            self.lim._sem.release()

    def slot(self, ip: str) -> _Ctx:
        return RateLimiter._Ctx(self, ip)

    @property
    def stats(self) -> dict:
        return {"active_ips": len(self._hist),
                "free": self._sem._value if self._sem else 0,
                "max_concurrent": self.max_concurrent}
