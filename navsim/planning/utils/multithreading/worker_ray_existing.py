"""
Ray worker that connects to existing Ray cluster without reinitializing.
"""
import logging
from concurrent.futures import Future
from pathlib import Path
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
        output_dir: Optional[Union[str, Path]] = None,
        logs_subdir: str = "logs",
        use_distributed: bool = False,
    ):
        """
        Initialize Ray worker for existing cluster.
        
        :param master_node_ip: Not used, kept for compatibility
        :param threads_per_node: Not used, kept for compatibility  
        :param debug_mode: If true, execute serially for debugging
        :param log_to_driver: If true, show logs from workers
        :param output_dir: Experiment output directory
        :param logs_subdir: Subdirectory for logs
        :param use_distributed: Not used, kept for compatibility
        """
        self._debug_mode = debug_mode
        self._log_to_driver = log_to_driver
        self._log_dir: Optional[Path] = Path(output_dir) / logs_subdir if output_dir is not None else None
        
        # Check if Ray is already initialized
        if not ray.is_initialized():
            logger.error("Ray is not initialized! Please start Ray first with: ray start --head")
            raise RuntimeError("Ray cluster not found. Please start Ray first.")
        
        logger.info("Connected to existing Ray cluster")
        
        # Get cluster resources
        cluster_resources = ray.cluster_resources()
        self._number_of_cpus = int(cluster_resources.get('CPU', cpu_count()))
        self._number_of_gpus = int(cluster_resources.get('GPU', 0))
        
        logger.info(f"Ray cluster resources - CPU: {self._number_of_cpus}, GPU: {self._number_of_gpus}")
        
        worker_resources = WorkerResources(
            number_of_nodes=1,  # Simplified assumption
            number_of_cpus_per_node=self._number_of_cpus,
            number_of_gpus_per_node=self._number_of_gpus,
        )
        
        # Initialize parent class with worker resources
        super().__init__(worker_resources)

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        Note: We don't shutdown Ray as it's an existing cluster.
        """
        logger.info("Disconnecting from Ray cluster (cluster remains running)")
        pass

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose  # Not used
        if self._debug_mode:
            logger.info("Running in debug mode (serial execution)")
            # Serial execution for debugging
            if len(item_lists) == 1:
                return [task.fn(item) for item in item_lists[0]]
            else:
                # Multiple argument lists
                import itertools
                return [task.fn(*items) for items in zip(*item_lists)]
        else:
            logger.info(f"Executing tasks on Ray cluster")
            return ray_map(task, *item_lists, log_dir=self._log_dir)

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        """Inherited, see superclass."""
        if self._debug_mode:
            # Execute synchronously for debugging
            future = Future()
            try:
                result = task.fn(*args, **kwargs)
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)
            return future
        else:
            # Submit to Ray
            remote_fn = ray.remote(task.fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
            object_ids: ray._raylet.ObjectRef = remote_fn.remote(*args, **kwargs)
            return object_ids.future()  # type: ignore