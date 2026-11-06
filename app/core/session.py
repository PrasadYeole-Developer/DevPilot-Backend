"""
Development session identity (Milestone 7).

DevPilot has no real authentication yet. Rather than pretending anonymous
requests are authenticated users (or silently trusting a client-supplied
user_id), this is an explicit, documented placeholder: callers identify
themselves via an `X-Session-Id` header. If omitted, the server mints a new
one and returns it on the response, so the same caller can reuse it on
subsequent requests to keep owning their conversations.

This is NOT authentication — anyone who learns a session ID can act as that
session. It only prevents *accidental* cross-conversation leakage between
different callers in normal use, and gives every conversation/memory a
stable owner key to replace once real auth exists. This should be replaced
by a real identity (e.g. from an auth middleware) in a future milestone.
"""

import uuid

from fastapi import Request, Response

SESSION_HEADER_NAME = "X-Session-Id"


def get_session_id(request: Request, response: Response) -> str:
    session_id = request.headers.get(SESSION_HEADER_NAME)
    if not session_id:
        session_id = uuid.uuid4().hex
    # Always echo it back, so a caller who didn't send one learns the ID the
    # server assigned them for this request.
    response.headers[SESSION_HEADER_NAME] = session_id
    return session_id
