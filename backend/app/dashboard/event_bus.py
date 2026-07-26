"""ASTER lightweight in-memory event bus."""

from __future__ import annotations

import asyncio
from typing import Any

# Global list of active subscriptions (queue, loop)
_subscribers: list[tuple[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop]] = []

def subscribe(q: asyncio.Queue[dict[str, Any]], loop: asyncio.AbstractEventLoop) -> None:
    """Subscribe a queue to the event bus."""
    _subscribers.append((q, loop))

def unsubscribe(q: asyncio.Queue[dict[str, Any]], loop: asyncio.AbstractEventLoop) -> None:
    """Unsubscribe a queue from the event bus."""
    if (q, loop) in _subscribers:
        _subscribers.remove((q, loop))

def broadcast(event: dict[str, Any]) -> None:
    """Broadcast an event to all subscribers from any thread."""
    for q, loop in _subscribers:
        if not loop.is_closed():
            loop.call_soon_threadsafe(q.put_nowait, event)
