"""Process-local CLOSED/OPEN/HALF_OPEN circuit, one probe during cooldown recovery."""
import threading
import time

class CircuitOpen(RuntimeError):
    pass

def classify(status=None, kind=None, error_type=None):
    if status in {401,403}: return 'authentication'
    if status==429: return 'rate_limit'
    if isinstance(status,int) and 500<=status<600: return 'server'
    if status is not None: return 'request'
    if kind and 'timeout' in kind.lower(): return 'timeout'
    if kind: return 'connection'
    if error_type=='BudgetExceeded': return 'budget'
    return 'schema'

class Circuit:
    def __init__(self, threshold=3, cooldown=30., clock=time.monotonic):
        if threshold<1 or cooldown<=0: raise ValueError('Invalid circuit configuration')
        self.threshold,self.cooldown,self.clock=threshold,cooldown,clock
        self.state='CLOSED'; self.failures=0; self.opened=0.; self.probing=False; self.generation=0
        self.lock=threading.Lock()

    def enter(self):
        return self.enter_ticket()[0]

    def enter_ticket(self):
        with self.lock:
            if self.state=='OPEN':
                if self.clock()-self.opened<self.cooldown: raise CircuitOpen('provider_circuit_open')
                self.state='HALF_OPEN'
            if self.state=='HALF_OPEN':
                if self.probing: raise CircuitOpen('provider_probe_in_progress')
                self.probing=True
            return self.state, self.generation

    def finish(self, category=None, ticket=None):
        with self.lock:
            # Older in-flight attempts cannot close a circuit opened by another
            # request, nor release a newer half-open probe.
            if ticket is not None and ticket != self.generation:
                return
            self.probing=False
            if category is None:
                if self.state == 'HALF_OPEN': self.generation += 1
                self.state='CLOSED'; self.failures=0
            elif category in {'authentication','rate_limit','server','timeout','connection'}:
                self.failures+=1
                if self.state=='HALF_OPEN' or self.failures>=self.threshold or category=='authentication':
                    self.state='OPEN'; self.opened=self.clock(); self.generation+=1
            elif category == 'budget':
                # A local deadline is not evidence that a half-open provider recovered.
                if self.state == 'HALF_OPEN':
                    self.state='OPEN'; self.opened=self.clock(); self.generation+=1
            else:
                # A 4xx or malformed response proves transport connectivity.
                if self.state == 'HALF_OPEN': self.generation+=1
                self.state='CLOSED'; self.failures=0

_registry={}
_lock=threading.Lock()
def circuit_for(key, threshold, cooldown):
    with _lock:
        return _registry.setdefault((key,threshold,cooldown),Circuit(threshold,cooldown))
