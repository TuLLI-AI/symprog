import warnings
from typing import Type, Optional, Any, Union, ClassVar, Generic, TypeVar

from .base import UtilBase
from .config import BaseConfig, FastConfig

__all__ = ["Registry"]

T = TypeVar("T", bound=UtilBase)

def _same_definition(cls1: type, cls2: type) -> bool:
    """Whether two classes come from the same definition (e.g., a re-run notebook cell),
    including the component class of automatically wrapped semirings."""
    def key(c):
        inner = getattr(c, "Leftsemiring", None)
        inner_key = (inner.__module__, inner.__qualname__) if inner is not None else None
        return (c.__module__, c.__qualname__, inner_key)
    return key(cls1) == key(cls2)

class Registry(UtilBase, Generic[T]):
    """General class decorator for registering new class implementations. 

    Allows passing around implementations across modules, as well as defining 
    new extensions outside of the library. 

    Attributes: 
        _IMPLEMENTATIONS: The storage of implementations. 

    Examples
    --------

    Below would be an example class register into `Registry` (in a notebook):
    
    Starting with 

    >>> import symprog
    >>> from symprog import FastConfig
    >>> from symprog import Registry 
    
    Then implement the following class 

    ```python 

       @Registry(name="MyClass", config=FastConfig.from_dict({"x" : 10}))
       class MyClass(cot_prog.base.UtilBase): 
           def __init__(self, x): 
               self.x = x

           @classmethod 
           def from_config(cls, config): 
               return cls(config)
    ```
    
    then do 

    >>> c = Registry.build_instance("MyClass", x=20)
    >>> c.x
    20 

    """
    _IMPLEMENTATIONS: dict[str, tuple[Type[T], Type[BaseConfig]]]

    def __init__(
        self, name: str, config: Optional[Type[BaseConfig]] = None
    ) -> None:
        """Creates a `Register` instance.

        Arguments:
            name: The unique identifier of the recourse being registered.
            config_cls: The (optional) configuration class tied with the implementation.
        Raises: 
            TypeError when wrong configuration class is passed. 
        """
        if not (config is None or issubclass(config, BaseConfig)):
            raise TypeError(f"Config must be a BaseConfig subclass or None, got {config!r}")

        self.name = name
        self.config_cls = config

    def __init_subclass__(cls, **kwargs: Any) -> None:
        ### ensures that implementations from parents doesn't carry over.
        super().__init_subclass__(**kwargs)
        cls._IMPLEMENTATIONS = {}

    def __call__(self, cls_: Type[T]) -> Type[T]:
        identifier = self.name
        if identifier in self._IMPLEMENTATIONS:
            existing_cls, _ = self._IMPLEMENTATIONS[identifier]
            if not _same_definition(existing_cls, cls_):
                warnings.warn(
                    f"Item already registered, Please manually remove: {identifier}"
                )
                return cls_
            ## re-running the same definition (e.g., a notebook cell) replaces it

        if not hasattr(cls_, "from_config"):
            raise TypeError(f"{cls_.__name__} must define from_config(cls, config)")


        self._IMPLEMENTATIONS[identifier] = (cls_, self.config_cls) # type: ignore 
        return cls_

    @classmethod
    def build_instance(cls, name: str, **kwargs) -> T:
        """Builds a class within `cls.IMPLEMENTATIONS` from configuration

        Arguments:
            config: The global config used for building an object.
        Raises:
            ValueError when requested item is not registered. 
        """
        implementation_tuple = cls._IMPLEMENTATIONS.get(name)
        if implementation_tuple is None:
            impl = list(cls._IMPLEMENTATIONS.keys())
            raise ValueError(
                f"Resource not found registered, value={name}, available classes={', '.join(impl)}"
            )
        
        impl_cls, config_cls = implementation_tuple
        if config_cls is None: 
            return impl_cls.from_config(**kwargs)
        
        config = config_cls.build_config(**kwargs)
        return impl_cls.from_config(config) # type: ignore 
