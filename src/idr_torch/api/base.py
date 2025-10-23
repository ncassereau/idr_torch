#! /usr/bin/env python

import socket
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from ..utils import _TORCH_AVAILABLE, IdrTorchWarning

if TYPE_CHECKING and _TORCH_AVAILABLE:
    import torch


def keep_as_func(func: Callable) -> Callable:
    func.__keep_as_func__ = True
    return func


def depends_on_torch(func: Callable) -> Callable:
    if _TORCH_AVAILABLE:
        return func
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError("This function requires torch which is not available")

        return wrapper


class API(ABC):
    priority: int = 5000
    name: str = "AbstractAPI"

    @abstractmethod
    def is_launcher(self) -> bool:
        """
        Detects if the given API is the one used to launch the current job.
        """
        raise NotImplementedError()

    @abstractmethod
    def rank(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def local_rank(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def world_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def local_world_size(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def cpus(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def gpus(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def nodelist(self) -> str | list[str]:
        raise NotImplementedError()

    @abstractmethod
    def master_address(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def port(self) -> int:
        raise NotImplementedError()

    def is_master(self) -> bool:
        return self.rank() == 0

    @depends_on_torch
    def device(self) -> "torch.device":
        import torch

        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank()}")
        else:
            return torch.device("cpu")

    @depends_on_torch
    @keep_as_func
    def init_process_group(
        self, *args, force_init: bool = False, **kwargs
    ) -> "torch.device":
        import torch.distributed as dist

        _kwargs = dict(rank=self.rank(), world_size=self.world_size())
        _kwargs.update(**kwargs)

        if dist.is_initialized():
            if force_init:
                warnings.warn(
                    message=(
                        "A distributed environment had already been initialized, "
                        "but you requested to force the initialization. Attempting "
                        "to destroy the process group before recreating it."
                    ),
                    category=IdrTorchWarning,
                    stacklevel=4,
                )
                dist.destroy_process_group()
                dist.init_process_group(*args, **_kwargs)
            else:
                warnings.warn(
                    message=(
                        "A distributed environment had already been initialized."
                        " Moving on."
                    ),
                    category=IdrTorchWarning,
                    stacklevel=4,
                )
        else:
            dist.init_process_group(*args, **_kwargs)
        return self.device()

    def hostname(self) -> str:
        return socket.gethostname()
