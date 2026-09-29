"""Langfuse tracing, opt-in via LANGFUSE_PUBLIC_KEY. Traces go to the self-hosted instance only."""
import os
from contextlib import contextmanager
from urllib.parse import urlparse


def enabled() -> bool:
    # Traces carry note excerpts, so only a local Langfuse counts. The SDK would otherwise
    # silently default to cloud.langfuse.com when no host is set.
    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL") or ""
    local = urlparse(host).hostname in ("127.0.0.1", "localhost", "::1")
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and local


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
