"""
Timeout and backpressure utilities for resilient LLM calling.

This module provides:
1. Request-level timeout wrapper for long-running operations
2. Graceful fallback handling when timeouts occur
3. Tracking of request execution time for backpressure signals
"""

import asyncio
import time
from typing import Callable, Any, Optional, Dict, TypeVar, Coroutine
from functools import wraps

T = TypeVar('T')

class TimeoutConfig:
    """Configuration for timeout behavior"""
    def __init__(
        self,
        default_timeout_ms: int = 5000,
        max_total_request_ms: int = 30000,
        warn_threshold_ms: int = 3000
    ):
        self.default_timeout_ms = default_timeout_ms
        self.max_total_request_ms = max_total_request_ms
        self.warn_threshold_ms = warn_threshold_ms


class RequestTimer:
    """Track request execution time and enforce timeouts"""
    def __init__(self, request_id: str, max_total_ms: int = 30000):
        self.request_id = request_id
        self.max_total_ms = max_total_ms
        self.start_time = time.time()
        self.stage_times: Dict[str, float] = {}
    
    def elapsed_ms(self) -> float:
        """Get total elapsed time since request start"""
        return (time.time() - self.start_time) * 1000
    
    def remaining_ms(self) -> float:
        """Get remaining time before hitting max budget"""
        return max(0, self.max_total_ms - self.elapsed_ms())
    
    def is_expired(self) -> bool:
        """Check if total request time exceeded"""
        return self.remaining_ms() <= 0
    
    def check_budget(self, stage: str, timeout_ms: int) -> bool:
        """Check if we have enough budget for next stage"""
        remaining = self.remaining_ms()
        is_tight = remaining < timeout_ms
        
        if is_tight:
            print(f"⏱️  [TIMEOUT] Request {self.request_id} stage '{stage}': "
                  f"only {remaining:.0f}ms remaining, need {timeout_ms}ms", flush=True)
        
        return not is_tight
    
    def record_stage(self, stage: str) -> float:
        \"\"\"Record completion of a stage and return elapsed time for this stage\"\"\"
        current_time = time.time()
        stage_elapsed = current_time - (
            self.start_time + sum(self.stage_times.values())
        ) * 1000 / 1000
        
        self.stage_times[stage] = current_time - self.start_time
        return stage_elapsed
    
    def get_summary(self) -> dict:
        \"\"\"Get timing summary\"\"\"
        return {
            "request_id": self.request_id,
            "total_elapsed_ms": self.elapsed_ms(),
            "max_budget_ms": self.max_total_ms,
            "remaining_ms": self.remaining_ms(),
            "stages": self.stage_times
        }


async def call_with_timeout(
    coro_or_func: Any,
    timeout_ms: int = 5000,
    fallback_result: Any = None,
    stage_name: str = "operation",
    request_timer: Optional[RequestTimer] = None,
    request_id: str = "unknown"
) -> Any:
    \"\"\"
    Call an async function or coroutine with timeout and fallback.
    
    Args:
        coro_or_func: Coroutine or async function to call
        timeout_ms: Timeout in milliseconds
        fallback_result: Result to return if timeout occurs
        stage_name: Name of stage for logging
        request_timer: Optional RequestTimer to check budget
        request_id: Request ID for logging
    
    Returns:
        Result from function or fallback_result on timeout
    \"\"\"
    # Check if we have time budget remaining
    if request_timer and not request_timer.check_budget(stage_name, timeout_ms):
        print(f"⚠️  [FALLBACK] Skipping {stage_name} - insufficient time budget", flush=True)
        return fallback_result
    
    timeout_sec = timeout_ms / 1000.0
    
    try:
        if asyncio.iscoroutine(coro_or_func):
            result = await asyncio.wait_for(coro_or_func, timeout=timeout_sec)
        else:
            # If it's a regular function, run it in executor
            result = await asyncio.wait_for(
                asyncio.to_thread(coro_or_func),
                timeout=timeout_sec
            )
        
        if request_timer:
            request_timer.record_stage(stage_name)
        
        return result
    
    except asyncio.TimeoutError:
        elapsed_ms = request_timer.elapsed_ms() if request_timer else timeout_ms
        print(f"⚠️  [TIMEOUT] {stage_name} timed out after {elapsed_ms:.0f}ms "
              f"(max: {timeout_ms}ms) - using fallback. RequestID: {request_id}", flush=True)
        return fallback_result
    
    except Exception as e:
        print(f"⚠️  [ERROR] {stage_name} failed with: {str(e)} - using fallback", flush=True)
        return fallback_result


def timeout_decorator(
    timeout_ms: int = 5000,
    fallback_return: Any = None
):
    \"\"\"
    Decorator to add timeout to an async function.
    
    Usage:
        @timeout_decorator(timeout_ms=5000, fallback_return={})
        async def my_function():
            ...
    \"\"\"
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await call_with_timeout(
                func(*args, **kwargs),
                timeout_ms=timeout_ms,
                fallback_result=fallback_return,
                stage_name=func.__name__
            )
        return wrapper
    return decorator


def create_llm_timeout_wrapper(
    llm_invoke_func: Callable,
    timeout_ms: int = 5000,
    request_id: str = "unknown"
) -> Callable:
    \"\"\"
    Create a wrapper function that calls an LLM's invoke method with timeout.
    
    This is useful for synchronous LLM calls that need timeout handling.
    
    Usage:
        wrapped_planning_llm = create_llm_timeout_wrapper(
            planning_llm.invoke,
            timeout_ms=5000,
            request_id=request_id
        )
        result = wrapped_planning_llm(query, data_source)
    \"\"\"
    @wraps(llm_invoke_func)
    def wrapper(*args, **kwargs):
        try:
            # Run the sync function with timeout
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.to_thread(llm_invoke_func, *args, **kwargs),
                        timeout=timeout_ms / 1000.0
                    )
                )
                return result
            finally:
                loop.close()
        except asyncio.TimeoutError:
            print(f"⚠️  [TIMEOUT] LLM call ({llm_invoke_func.__qualname__}) timed out "
                  f"after {timeout_ms}ms (RequestID: {request_id})", flush=True)
            return {"success": False, "error": f"Timeout after {timeout_ms}ms", "timed_out": True}
        except Exception as e:
            print(f"⚠️  [ERROR] LLM call ({llm_invoke_func.__qualname__}) failed: {str(e)}", flush=True)
            return {"success": False, "error": str(e), "exception": True}
    
    return wrapper


# Global timeout configuration
DEFAULT_TIMEOUT_CONFIG = TimeoutConfig(
    default_timeout_ms=5000,
    max_total_request_ms=30000,
    warn_threshold_ms=3000
)
