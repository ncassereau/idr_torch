"""Type stubs for idr_torch module.

This file provides type hints for Pylance and other type checkers.
At runtime, the module is replaced by an Interface instance that dynamically
routes to the appropriate API implementation.
"""

import torch

from .api import API as API
from .api import AutoMasterAddressPort as AutoMasterAddressPort
from .api import decorate_methods as decorate_methods
from .api import modifiers as modifiers
from .utils import IdrTorchWarning as IdrTorchWarning

__version__: str

# Properties - these are accessible as attributes, not methods
rank: int
"""Property containing the rank of the process."""

local_rank: int
"""Property containing the local rank of the process."""

world_size: int
"""Property containing the number of processes launched."""

local_world_size: int
"""Property containing the number of processes launched of each node."""

num_nodes: int
"""Property containing the number of nodes."""

cpus: int
"""Property containing the number of CPUs allocated to each process."""

gpu_ids: list[str]
"""Property containing all GPUs ids."""

nodelist: str | list[str]
"""Property containing the list of nodes."""

master_addr: str
"""Property containing the master node."""

master_port: int
"""Property containing the port to communicate with the master process."""

is_master: bool
"""Detects whether the process is the master (i.e. the rank 0)."""

device: torch.device
"""Returns the torch device for this process."""

hostname: str
"""Returns the hostname of the current node."""

# Functions (methods marked with @keep_as_func)
def init_process_group(*args, force_init: bool = False, **kwargs) -> torch.device:
    """
    Initialize the distributed process group.

    See https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group
    for more information. Also returns the device.

    Args:
        *args: Positional arguments passed to torch.distributed.init_process_group
        force_init: If True, destroy existing process group before reinitializing
        **kwargs: Additional keyword arguments passed to
            torch.distributed.init_process_group

    Returns:
        torch.device: The device for this process
    """
    ...

# Aliases for properties
ntasks: int
"""Alias for world_size."""

size: int
"""Alias for world_size."""

local_size: int
"""Alias for local_world_size."""

ntasks_per_node: int
"""Alias for local_world_size."""

nnodes: int
"""Alias for num_nodes."""

cpus_per_task: int
"""Alias for cpus."""

master_address: str
"""Alias for master_addr."""

# Aliases for init_process_group
def init_pg(*args, force_init: bool = False, **kwargs) -> torch.device:
    """Alias for init_process_group."""
    ...

def init(*args, force_init: bool = False, **kwargs) -> torch.device:
    """Alias for init_process_group."""
    ...

# Module-level utilities
def register_API(new_API: API) -> None:
    """Register a new API implementation."""
    ...

def get_launcher_API() -> API:
    """Get the currently active launcher API."""
    ...

current_API: str
"""Name of the currently active API."""

all_APIs: list[API]
"""List of all registered API implementations."""

def crawl_module_for_APIs(module) -> None:
    """Crawl a module to find and register API implementations."""
    ...

def summary(tab_length: int = 4) -> str:
    """Print a summary of the current distributed configuration."""
    ...
