"""Portable CUDA failure contract; importing this module never imports torch."""
from contextlib import contextmanager
import functools
import sys
import threading
import traceback


_predict_cache_lock = threading.RLock()


class CudaOutOfMemoryError(RuntimeError):
    error_kind = 'cuda_out_of_memory'


def is_cuda_oom(exc):
    if isinstance(exc, CudaOutOfMemoryError):
        return True
    torch = sys.modules.get('torch')
    cuda = getattr(torch, 'cuda', None)
    oom = getattr(cuda, 'OutOfMemoryError', None)
    if isinstance(oom, type) and isinstance(exc, oom):
        return True
    # Older CUDA allocators use RuntimeError. Host MemoryError and generic
    # "out of memory" messages are deliberately not sufficient evidence.
    return isinstance(exc, RuntimeError) and 'cuda out of memory' in str(exc).lower()


def release_exception(exc):
    """Clear unwound tensor-bearing frames, including chained native errors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc.__traceback__ = None
        other = exc.__cause__ or exc.__context__
        exc.__cause__ = exc.__context__ = None
        exc = other


def observe_forward(call, observed):
    """Retain classification, never exception/tensor references swallowed upstream."""
    @functools.wraps(call)
    def forward(*args, **kwargs):
        try:
            return call(*args, **kwargs)
        except Exception as exc:
            if is_cuda_oom(exc):
                observed['cuda_out_of_memory'] = True
            raise
    return forward


@contextmanager
def observe_predict_oom(cuda, observed):
    """Observe Boltz's swallowed RuntimeError at its existing cache cleanup.

    Boltz 2.2.1 calls empty_cache inside the predict_step RuntimeError handler,
    including failures before forward and in post-forward confidence arithmetic.
    Inspect only the active exception at that call; retain only a boolean. There
    is no profiler, source rewrite, tensor operation, or normal-forward callback.
    Serialize our temporary overrides and ignore unrelated threads' exceptions.
    """
    with _predict_cache_lock:
        original = cuda.empty_cache
        owner = threading.get_ident()

        @functools.wraps(original)
        def empty_cache(*args, **kwargs):
            if threading.get_ident() == owner and is_cuda_oom(sys.exc_info()[1]):
                observed['cuda_out_of_memory'] = True
            return original(*args, **kwargs)

        cuda.empty_cache = empty_cache
        try:
            yield
        finally:
            cuda.empty_cache = original


def native_boundary(call):
    """Translate only real CUDA OOMs at every public native boundary."""
    @functools.wraps(call)
    def wrapped(self, *args, **kwargs):
        oom = False
        attempt = args[1] if len(args) > 1 else kwargs.get('attempt')
        try:
            return call(self, *args, **kwargs)
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            oom = True
            release_exception(exc)
        finally:
            if oom:
                if self.benchmark_metrics is None:
                    self.benchmark_metrics = {'allocator_peak_reset': False,
                                              'allocator_peak_scope': 'since_last_reset_not_cell_isolated'}
                self.benchmark_metrics.update(error_kind='cuda_out_of_memory', failure_boundary=call.__name__)
            if call.__name__ == 'predict':
                # Also covers transfer/capture failures before hook registration.
                prepared = args[0] if args else kwargs.get('prepared')
                if isinstance(prepared, dict):
                    prepared['batch'] = None
                self.failure_metrics(attempt)
            elif oom:
                self.failure_metrics(attempt)
        if oom:
            raise CudaOutOfMemoryError('CUDA allocator exhausted') from None
    return wrapped
