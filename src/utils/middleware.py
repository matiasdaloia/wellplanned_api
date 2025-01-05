import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from .logger import get_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Get user ID if available
        user_id = None
        if hasattr(request.state, "user"):
            user_id = request.state.user.get("id")

        # Create logger with request context
        log = get_logger(request_id=request_id, user_id=user_id)

        # Start timer
        start_time = time.time()

        # Log request
        log.info(
            f"Incoming request {request.method} {request.url.path}",
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": str(request.query_params),
                    "client_host": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                }
            },
        )

        try:
            # Process request
            response = await call_next(request)
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            log.info(
                f"Request completed {request.method} {request.url.path}",
                extra={
                    "extra_data": {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                    }
                },
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log.error(
                f"Request failed {request.method} {request.url.path}",
                exc_info=True,
                extra={
                    "extra_data": {
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": duration_ms,
                        "error": str(e),
                    }
                },
            )
            raise


def setup_middleware(app: FastAPI) -> None:
    """Set up all middleware for the application"""
    app.add_middleware(RequestLoggingMiddleware)
