"""
Async Executor for CPU-intensive ML operations

Provides a clean interface to run blocking operations in thread pools
without blocking the async event loop.

Usage:
    executor = AsyncMLExecutor(max_workers=2)
    result = await executor.run(blocking_function, arg1, arg2, kwarg1=value)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, Optional
from functools import partial

logger = logging.getLogger(__name__)


class AsyncMLExecutor:
    """
    Thread pool executor for CPU-intensive ML operations.
    
    This allows blocking operations (like PyTorch inference) to run in
    separate threads without blocking the async event loop.
    
    Benefits:
    - Non-blocking predictions
    - Better concurrency
    - Ctrl+C works (main thread remains responsive)
    - Easy to extend for other CPU-intensive operations
    """
    
    def __init__(self, max_workers: int = 2, name: str = "ml-executor"):
        """
        Initialize the executor.
        
        Args:
            max_workers: Maximum number of worker threads
            name: Name for logging purposes
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=name
        )
        self.name = name
        self.max_workers = max_workers
        logger.info(f"AsyncMLExecutor '{name}' initialized with {max_workers} workers")
    
    async def run(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Run a blocking function in a thread pool.
        
        Args:
            func: The blocking function to run
            *args: Positional arguments for the function
            timeout: Optional timeout in seconds
            **kwargs: Keyword arguments for the function
        
        Returns:
            The result of the function
        
        Raises:
            asyncio.TimeoutError: If the operation times out
            Exception: Any exception raised by the function
        """
        loop = asyncio.get_event_loop()
        
        # Create a partial function with all arguments bound
        bound_func = partial(func, *args, **kwargs)
        
        try:
            # Run in thread pool with optional timeout
            if timeout:
                result = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, bound_func),
                    timeout=timeout
                )
            else:
                result = await loop.run_in_executor(self.executor, bound_func)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {timeout}s in {self.name}")
            raise
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            raise
    
    async def run_batch(
        self,
        func: Callable,
        items: list,
        timeout: Optional[float] = None
    ) -> list:
        """
        Run a function on multiple items concurrently.
        
        Args:
            func: The function to run (should accept a single item)
            items: List of items to process
            timeout: Optional timeout per item in seconds
        
        Returns:
            List of results in the same order as items
        """
        tasks = [
            self.run(func, item, timeout=timeout)
            for item in items
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the executor.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info(f"Shutting down {self.name} (wait={wait})")
        self.executor.shutdown(wait=wait)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


# Global executor instance (can be shared across the application)
_global_executor: Optional[AsyncMLExecutor] = None


def get_global_executor(max_workers: int = 2) -> AsyncMLExecutor:
    """
    Get or create the global ML executor.
    
    This is a singleton pattern to avoid creating multiple thread pools.
    
    Args:
        max_workers: Maximum number of worker threads (only used on first call)
    
    Returns:
        The global AsyncMLExecutor instance
    """
    global _global_executor
    
    if _global_executor is None:
        _global_executor = AsyncMLExecutor(
            max_workers=max_workers,
            name="global-ml-executor"
        )
    
    return _global_executor


def shutdown_global_executor(wait: bool = True):
    """
    Shutdown the global executor.
    
    Args:
        wait: Whether to wait for pending tasks to complete
    """
    global _global_executor
    
    if _global_executor is not None:
        _global_executor.shutdown(wait=wait)
        _global_executor = None

