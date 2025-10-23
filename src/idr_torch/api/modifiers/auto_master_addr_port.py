import os
from functools import wraps

from .. import API
from .decorate_methods import decorate_methods

env_variables_set: bool = False


def set_master_addr_port_env_variables(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        global env_variables_set
        if not env_variables_set:
            # must be done before actually setting the variable to prevent stackoverflow
            env_variables_set = True

            os.environ["MASTER_ADDR"] = self.master_address()
            os.environ["MASTER_PORT"] = str(self.port())
            os.environ["RANK"] = str(self.rank())
            os.environ["LOCAL_RANK"] = str(self.local_rank())
            os.environ["WORLD_SIZE"] = str(self.world_size())
            os.environ["LOCAL_WORLD_SIZE"] = str(self.local_world_size())
        return func(self, *args, **kwargs)

    return wrapper


def AutoMasterAddressPort(cls: type[API]) -> type[API]:
    return decorate_methods(cls, func_to_apply=set_master_addr_port_env_variables)
