import json
import sys
import inspect
from abc import abstractmethod
from typing import TypeVar, Optional, Any, Type
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass, fields, asdict
from argparse import ArgumentParser, BooleanOptionalAction

C = TypeVar("C", bound="BaseConfig")

__all__ = ["UtilityConfig", "FastConfig"]

class BaseConfig:
    """Base class for configuration objects.

    Configuration objects are used throughout for instantiating objects of all kinds.
    """

    def to_dict(self) -> dict[str, Any]:
        """Maps configuration to a dictionary."""
        raise NotImplementedError

    def dump(self, path: str) -> None:
        """Dumps the configuration to file

        Arguments:
            path: The path to dump configuration to.
        """
        with open(path, "w") as config_out:
            config_out.write(json.dumps(self.to_dict(), indent=4))

    @classmethod
    def load_from_file(cls: Type[C], file_path: str) -> C:
        """Loads a configuration from file.

        Arguments:
            file_path: The path to the file containing configuration
                values.
        """
        with open(file_path) as config_in:
            config_json = json.loads(config_in.read())
            return cls(**config_json)

    @classmethod
    def build_parser(cls, **kwargs) -> ArgumentParser:
        """Builds

        Arguments:
            argv: The command line input.
        """
        parser = ArgumentParser()
        return cls._populate_parser(parser, **kwargs)

    @classmethod
    def _populate_parser(cls: Type[C], parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Adds parameters to an argument parser base on underlyinh configuration class.

        Arguments:
            parser: The input argument parser.
        """
        raise NotImplementedError

    @classmethod
    def parse_cli(cls: Type[C], argv: Optional[list[str]] = sys.argv) -> C:
        """Creates configuration instance from CLI arguments. Automatically
        builds corresponding argparse parser.

        Arguments:
            argv: The raw CLI input.
        """
        parser = cls.build_parser()
        config = parser.parse_args(argv)

        return cls(**vars(config))

    @classmethod
    def from_kwargs(cls: Type[C], **kwargs: Any) -> C:
        """Build config from direct kwargs only.

        Arguments:
            kwargs: The input kwargs.
        """
        return cls(**kwargs)

    @classmethod
    def build_config(
        cls: Type[C],
        argv: Optional[list[str]] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> C:
        """Builds a configuration instance, either from CLI input
        or direct arguments.

        Arguments:
            argv: (Optional) command line configuration values.
        Returns:
            A configuration object with the values based on either CLI
                input, an external file or direct keyword args (in that order).
        """
        if argv is not None:
            return cls.parse_cli(argv)
        elif config_path is not None:
            return cls.load_from_file(config_path)
        return cls.from_kwargs(**kwargs)


class UtilityConfig(BaseModel, BaseConfig):
    """A Pydantic configuration implementation.

    This kind of verification is most useful for complex configurations
    that might involve certain types of pydantic-style verification.
    """

    model_config = ConfigDict(protected_namespaces=())

    def to_dict(self) -> dict[str, Any]:
        """Maps configuration to a dictionary.

        Returns:
            Returns a string indexed dictionary representation of the
              configuration.
        """
        return self.model_dump()

    # @classmethod
    # def _populate_parser(
    #     cls: Type[C], parser: ArgumentParser, **kwargs
    # ) -> Type[ArgumentParser]:
    #     """Adds parameters to an argument parser base on underlyinh configuration class.

    #     Arguments:
    #         parser: The input argument parser.
    #     """
    #     for field_name, field in cls.model_fields.items(): # type: ignore 
    #         type_annot = None
    #         if inspect.isclass(field.annotation):
    #             type_annot = field.annotation

    #         args = {
    #             "help": field.description,
    #             "type": type_annot,
    #             "required": field.is_required(),
    #             "default": field.default,
    #         }

    #         if isinstance(field.default, bool):
    #             args.pop("type", None)
    #             # args["type"] = bool
    #             args["required"] = False
    #             args["action"] = BooleanOptionalAction

    #         parser.add_argument(f"--{field_name}", **args)

    #     return parser


class DataclassMeta(type):
    """Metaclass for avoiding needing to use dataclass decorator explicitly when building config"""

    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)
        return dataclass(cls) # type: ignore 


class FastConfig(BaseConfig, metaclass=DataclassMeta):
    """An alternative configuration based instead on dataclasses. Should be used for
    lower-level object configurations.

    NOTE: Do not use a dataclass decorator with this class, it is applied automatically.
    """

    @classmethod
    def _populate_parser(cls, parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Adds parameters to an argument parser base on underlying
        configuration class.

        Arguments:
            parser: The input argument parser.
        """
        for field in fields(cls): # type: ignore 

            type_annot = None
            if inspect.isclass(field.type):
                type_annot = field.type

            args = {"type": type_annot, "default": field.default}

            if isinstance(field.type, bool):
                args["type"] = bool
                args["required"] = None
                args["action"] = BooleanOptionalAction

            parser.add_argument(f"--{field.name}", **args)

        return parser

    def to_dict(self) -> dict[str, Any]:
        """Maps configuration to a dictionary.

        Returns:
            Returns a string indexed dictionary representation of the
              configuration.
        """
        return asdict(self) # type: ignore 

    @classmethod
    def from_dict(
        cls: Type[C],
        name: Optional[str] = "config",
        data: Optional[dict[str, Any]] = None,
    ) -> Type[C]:
        """Create new FastConfig subclass from a dictionary.

        Each key becomes a field; its type is inferred via `type(value)`,
        and its default is `value`.

        Arguments:
            name: The name of the new configuration class.
            data: Key/value pairs to turn into fields.

        For example:

        >>> from symprog import FastConfig
        >>> config = FastConfig.from_dict("new", {"a" : 10})()
        >>> config.a
        10
        """
        if data is None:
            data = {}
        annotations: dict[str, type] = {}
        namespace: dict[str, Any] = {}

        for key, value in data.items():
            annotations[key] = type(value)
            namespace[key] = value

        namespace["__annotations__"] = annotations

        meta = type(cls)
        DynamicConfig = meta(name, (cls,), namespace)
        return DynamicConfig
