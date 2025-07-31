"""
Ray worker that connects to existing Ray cluster without reinitializing.
"""
import logging
from concurrent.futures import Future
from typing import Any, Iterable, List, Optional, Union

import ray
from psutil import cpu_count

from nuplan.planning.utils.multithreading.ray_execution import ray_map
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool, WorkerResources

logger = logging.getLogger(__name__)

class RayExistingCluster(WorkerPool):
    """
    Ray worker that connects to an existing Ray cluster without reinitializing.
    """

    def __init__(
        self,
        master_node_ip: Optional[str] = None,
        threads_per_node: Optional[int] = None,
        debug_mode: bool = False,
        log_to_driver: bool = True,
        logs_subdir: str = "logs",
        use_distributed: bool = False,
    ):
        """
        Initialize Ray worker for existing cluster.
        
        :param master_node_ip: Not used, kept for compatibility
        :param threads_per_node: Not used, kept for compatibility  
        :param debug_mode: If true, execute serially for debugging
        :param log_to_driver: If true, show logs from workers
        :param logs_subdir: Subdirectory for logs
        :param use_distributed: Not used, kept for compatibility
        """
        self._debug_mode = debug_mode
        self._log_to_driver = log_to_driver
        
        super().__init__(self.initialize())

    def initialize(self) -> WorkerResources:
        """
        Initialize connection to existing Ray cluster and return worker resources.
        :return: WorkerResources describing the cluster.
        """
        # Check if Ray is already initialized
        if not ray.is_initialized():
            logger.error("Ray is not initialized! Please start Ray first with: ray start --head")
            raise RuntimeError("Ray cluster not found. Please start Ray first.")
        
        logger.info("Connected to existing Ray cluster")
        
        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        number_of_cpus = int(cluster_resources.get('CPU', cpu_count()))
        number_of_gpus = int(cluster_resources.get('GPU', 0))
        
        logger.info(f"Ray cluster resources - CPU: {number_of_cpus}, GPU: {number_of_gpus}")
        
        return WorkerResources(
            number_of_nodes=1,  # Simplified assumption for existing cluster
            number_of_cpus_per_node=number_of_cpus,
            number_of_gpus_per_node=number_of_gpus,
        )

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose  # Not used
        if self._debug_mode:
            logger.info("Running in debug mode (serial execution)")
            # Execute serially for debugging
            if len(item_lists) == 1:
                return [task.fn(arg) for arg in item_lists[0]]
            else:
                return [task.fn(*args) for args in zip(*item_lists)]
        else:
            logger.info(f"Executing {len(item_lists[0])} tasks on Ray cluster")
            return ray_map(task, *item_lists)

    def map(self, func: Any, argument_list: List[Any]) -> List[Any]:
        """Inherited, see superclass."""
        if self._debug_mode:
            logger.info("Running in debug mode (serial execution)")
            return [func(arg) for arg in argument_list]
        else:
            logger.info(f"Executing {len(argument_list)} tasks on Ray cluster")
            return ray_map(func, argument_list)

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future:
        """Inherited, see superclass."""
        if self._debug_mode:
            # Execute synchronously for debugging
            future = Future()
            try:
                result = task(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        else:
            # Submit to Ray
            ray_future = ray.remote(task).remote(*args, **kwargs)
            # Convert Ray future to standard Future
            future = Future()
            
            def _get_result():
                try:
                    result = ray.get(ray_future)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            
            # Start background thread to get result
            import threading
            threading.Thread(target=_get_result, daemon=True).start()
            return future

    def __enter__(self):
        """Inherited, see superclass."""
        return self

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        Note: We don't shutdown Ray as it's an existing cluster.
        """
        logger.info("Worker shutdown (Ray cluster remains running)")
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Inherited, see superclass."""
        # Don't shutdown Ray as it's an existing cluster
        logger.info("Disconnecting from Ray cluster (cluster remains running)")
        self.shutdown()