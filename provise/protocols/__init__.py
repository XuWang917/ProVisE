def create_protocol(name, config=None):
    from .registry import create_protocol as _create_protocol

    return _create_protocol(name, config)


def list_protocols():
    from .registry import list_protocols as _list_protocols

    return _list_protocols()


__all__ = ["create_protocol", "list_protocols"]
