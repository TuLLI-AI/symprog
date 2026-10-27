import logging
from abc import ABC
from typing import TypeVar, Type

C = TypeVar("C", bound="UtilBase")

class UtilBase(ABC):
    """Library base class"""

    @property
    def logging(self) -> logging.RootLogger:
        """Returns an instance logger 
        
        :rtype: logging.RootLogger
        """
        return logging.getLogger(f"{__name__}.{type(self).__name__}") # type: ignore

    @classmethod
    def from_config(cls: Type[C], **kwargs) -> C:
        """Builds an instance from configuration 

        Arguments: 
            config: The incoming configuration. 
        """
        return cls(**kwargs)

class ExperimentalWarning(UserWarning):
    """Warning for experimental functionality (e.g., tensorization, the JAX backend), 
    whose behavior and API may change. Can be silenced with 
    `warnings.filterwarnings("ignore", category=symprog.ExperimentalWarning)`."""
