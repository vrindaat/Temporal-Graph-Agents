import time
import functools
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict


@dataclass
class TimingRecord:
    function: str
    latency_ms: float
    metadata: dict = field(default_factory=dict)


class PerformanceTracker:
    def __init__(self):
        self.records: List[TimingRecord] = []

    def track(self, func_name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                self.records.append(TimingRecord(
                    function=func_name or func.__name__,
                    latency_ms=elapsed,
                ))
                return result
            return wrapper
        return decorator

    def record(self, function: str, latency_ms: float, **metadata):
        self.records.append(TimingRecord(function=function, latency_ms=latency_ms, metadata=metadata))

    def summary(self) -> dict:
        groups = defaultdict(list)
        for r in self.records:
            groups[r.function].append(r.latency_ms)
        return {
            name: {
                "count": len(times),
                "mean_ms": round(sum(times) / len(times), 1),
                "min_ms": round(min(times), 1),
                "max_ms": round(max(times), 1),
                "total_ms": round(sum(times), 1),
            }
            for name, times in groups.items()
        }

    def reset(self):
        self.records = []


tracker = PerformanceTracker()
