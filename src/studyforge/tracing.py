"""Langfuse tracing, opt-in via LANGFUSE_PUBLIC_KEY. Traces go to the self-hosted instance only."""
import os
import socket
from contextlib import contextmanager
from functools import lru_cache
from urllib.parse import urlparse


def enabled() -> bool:
    # Traces carry note excerpts, so only a local Langfuse counts. The SDK would otherwise
    # silently default to cloud.langfuse.com when no host is set.
    url = urlparse(os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or "")
    local = url.hostname in ("127.0.0.1", "localhost", "::1")
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and local and _reachable(url.hostname, url.port or 80)


# ponytail: checked once per process; start Langfuse before the app to get traces
@lru_cache
def _reachable(host: str, port: int) -> bool:
    """Keys in .env but Langfuse stopped: skip tracing instead of retrying every span export."""
    try:
        socket.create_connection((host, port), timeout=0.5).close()
        return True
    except OSError:
        return False


def callbacks() -> list:
    """LangChain callbacks: model, tokens, cost and the graph's agent/tool tree come from the integration."""
    if not enabled():
        return []
    from langfuse.langchain import CallbackHandler
    return [CallbackHandler()]


class _NoOp:
    def update(self, **kwargs) -> None:
        pass


@contextmanager
def observe(name: str, as_type: str, input=None, **trace_attrs):
    """A Langfuse observation (no-op when tracing is off). trace_attrs (session_id, tags, ...) propagate to children."""
    if not enabled():
        yield _NoOp()
        return
    import langfuse
    with langfuse.get_client().start_as_current_observation(name=name, as_type=as_type, input=input) as obs:
        if trace_attrs:
            with langfuse.propagate_attributes(**trace_attrs):
                yield obs
        else:
            yield obs


def flush() -> None:
    """Send queued events; call before a short-lived process exits."""
    if enabled():
        import langfuse
        langfuse.get_client().flush()
