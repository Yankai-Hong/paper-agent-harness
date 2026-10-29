"""Cooperative per-turn accounting; native calls are not forcibly interrupted."""
from contextvars import ContextVar
from dataclasses import dataclass, field
import time

current_budget = ContextVar('paper_turn_budget', default=None)

class BudgetExceeded(RuntimeError):
    pass

@dataclass
class TurnBudget:
    seconds: float = 180
    model_requests: int = 8
    subqueries: int = 6
    candidates: int = 90
    rerank: int = 60
    clock: object = time.monotonic
    used: dict = field(default_factory=dict)

    def __post_init__(self):
        if min(self.seconds,self.model_requests,self.subqueries,self.candidates,self.rerank) <= 0:
            raise ValueError('Turn budgets must be positive')
        self.started = self.clock()

    def remaining(self):
        remaining = self.seconds - (self.clock()-self.started)
        if remaining <= 0:
            raise BudgetExceeded('turn_deadline_exhausted')
        return remaining

    def take(self, kind, count=1):
        self.remaining()
        if count < 0 or self.used.get(kind,0)+count > getattr(self,kind):
            raise BudgetExceeded('turn_'+kind+'_exhausted')
        self.used[kind] = self.used.get(kind,0)+count

    def available(self, kind):
        self.remaining()
        return getattr(self,kind)-self.used.get(kind,0)
