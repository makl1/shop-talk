"""
Patches heavy ML and external dependencies into sys.modules before any test
imports services/api/main.py. This prevents model downloads, GPU usage, and
live network calls during testing.
"""
import sys
import os
from unittest.mock import MagicMock

# Make services/api importable as 'main'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "api"))

# ---------------------------------------------------------------------------
# tenacity — make @retry a transparent passthrough
# ---------------------------------------------------------------------------
_tenacity = MagicMock()
_tenacity.retry.side_effect = lambda **kw: (lambda fn: fn)
_tenacity.stop_after_attempt.return_value = None
_tenacity.wait_exponential.return_value = None
sys.modules["tenacity"] = _tenacity

# ---------------------------------------------------------------------------
# torch
# ---------------------------------------------------------------------------
_torch = MagicMock()
_torch.cuda.is_available.return_value = False
_ctx = MagicMock()
_ctx.__enter__ = lambda s: None
_ctx.__exit__ = lambda *a: False
_torch.no_grad.return_value = _ctx
_torch.Tensor = type("Tensor", (), {})  # real class so isinstance checks work
sys.modules["torch"] = _torch

# ---------------------------------------------------------------------------
# transformers
# ---------------------------------------------------------------------------
_transformers = MagicMock()
sys.modules["transformers"] = _transformers

# ---------------------------------------------------------------------------
# qdrant_client
# ---------------------------------------------------------------------------
sys.modules["qdrant_client"] = MagicMock()

# ---------------------------------------------------------------------------
# boto3 / botocore
# ---------------------------------------------------------------------------
sys.modules["boto3"] = MagicMock()
sys.modules["botocore"] = MagicMock()
sys.modules["botocore.config"] = MagicMock()

# ---------------------------------------------------------------------------
# prometheus_client — needs working Counter/Histogram context managers
# ---------------------------------------------------------------------------
_prom = MagicMock()

_counter_inst = MagicMock()
_counter_inst.labels.return_value = MagicMock()
_prom.Counter.return_value = _counter_inst

_time_ctx = MagicMock()
_time_ctx.__enter__ = lambda s: None
_time_ctx.__exit__ = lambda *a: False
_hist_inst = MagicMock()
_hist_inst.labels.return_value.time.return_value = _time_ctx
_prom.Histogram.return_value = _hist_inst

sys.modules["prometheus_client"] = _prom

# ---------------------------------------------------------------------------
# slowapi — Limiter.limit() must return the original function unchanged
# ---------------------------------------------------------------------------
class _FakeLimiter:
    def __init__(self, key_func=None):
        pass

    def limit(self, limit_str):
        return lambda fn: fn


_slowapi = MagicMock()
_slowapi.Limiter = _FakeLimiter
_slowapi._rate_limit_exceeded_handler = MagicMock()

_slowapi_errors = MagicMock()
_slowapi_errors.RateLimitExceeded = Exception

sys.modules["slowapi"] = _slowapi
sys.modules["slowapi.util"] = MagicMock()
sys.modules["slowapi.errors"] = _slowapi_errors
