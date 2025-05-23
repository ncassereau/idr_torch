#! /usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
from collections.abc import Iterable
from functools import wraps
from importlib.metadata import version
from inspect import isclass
from pathlib import Path
from typing import Any, List, Union

from . import __name__, __path__
from .api import API, AutoMasterAddressPort, DefaultAPI, decorate_methods
from .utils import IdrTorchWarning, warning_filter

__version__ = version(__name__)


class EmptyClass(object):
    pass


class Interface(object):
    def __init__(self):
        self._available_APIs: List[API] = []
        self.crawl_shipped_APIs()
        self.add_other_object_for_easy_access()
        self.add_API_functions()
        self.make_dir()

    @classmethod
    def add_attribute(cls, name, attribute) -> None:
        setattr(cls, name, attribute)

    def add_API_functions(self) -> None:
        from . import config

        for method_name in config.__all__:
            base_function = getattr(config, method_name)
            new_function = self.make_new_function(
                base_function.__name__,
                as_property=not getattr(base_function, "__keep_as_func__", False),
            )
            self.add_attribute(method_name, new_function)

    def make_dir(self):
        from . import config

        self.__dir: List[str] = []
        self.__dir += dir(EmptyClass())
        self.__dir += config.__all__
        self.__dir += self.__all__
        self.__dir += [
            "__version__",
        ]

    def crawl_shipped_APIs(self) -> None:
        from . import api

        self.crawl_module_for_APIs(api)

    def add_other_object_for_easy_access(self) -> None:
        from . import api
        from .api import modifiers

        self.api = api
        self.API = API
        self.AutoMasterAddressPort = AutoMasterAddressPort
        self.decorate_methods = decorate_methods
        self.IdrTorchWarning = IdrTorchWarning
        self.modifiers = modifiers
        self.__file__ = str(Path(__file__).parent / "__init__.py")
        self.__path__ = __path__
        self.__name__ = __name__
        self.__version__ = __version__
        self.__spec__ = __spec__
        self.__all__ = [
            "api",
            "API",
            "AutoMasterAddressPort",
            "decorate_methods",
            "IdrTorchWarning",
            "modifiers",
            "register_API",
            "get_launcher_API",
            "current_API",
            "all_APIs",
            "crawl_module_for_APIs",
            "summary",
        ]

    def __repr__(self) -> str:
        return f"<module '{self.__name__}' from '{self.__file__}'"

    def __dir__(self) -> Iterable[str]:
        return self.__dir

    def make_new_function(
        self, dest_name: str, /, as_property: bool = True
    ) -> Union[property, callable]:
        @wraps(getattr(API, dest_name))
        def redirect(self: Interface, *args, **kwargs) -> Any:
            with warnings.catch_warnings(record=True) as warning_list:
                api = self.get_launcher_API()
                output = getattr(api, dest_name)(*args, **kwargs)
            if warning_list:
                warning_filter.warn(warning_list)
            return output

        if as_property:
            return property(redirect)
        else:
            return redirect

    def register_API(self, new_API: API) -> None:
        for i, api in enumerate(self._available_APIs):
            if api.priority > new_API.priority:
                continue
            else:
                self._available_APIs.insert(i, new_API)
                break
        else:
            self._available_APIs.append(new_API)

    def get_launcher_API(self) -> API:
        for api in self._available_APIs:
            if api.is_launcher():
                return api
        return DefaultAPI()

    @property
    def current_API(self) -> str:
        return self.get_launcher_API().name

    @property
    def all_APIs(self) -> List[API]:
        return self._available_APIs

    def crawl_module_for_APIs(self, module) -> None:
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isclass(obj) and issubclass(obj, API) and obj is not API:
                # obj is the class so we instanciate it
                self.register_API(obj())
            elif isinstance(obj, API) and obj.__class__ is not API:
                # obj is already the instance
                self.register_API(obj)

    def summary_str(self, /, tab_length: int = 4) -> str:
        string = f"{self.current_API}(\n"
        values = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "cpus_per_task": self.cpus,
            "nodelist": self.nodelist,
            "hostname": self.hostname,
            "master_address": self.master_addr,
            "master_port": self.master_port,
        }
        for key, value in values.items():
            string += " " * tab_length + f"{key}={value},\n"
        string += ")"
        return string

    def summary(self, /, tab_length: int = 4) -> str:
        with warnings.catch_warnings(action="ignore", category=IdrTorchWarning):
            print(self.summary_str(tab_length=tab_length))
