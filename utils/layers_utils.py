class DummyContextMgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def make_layer(layer):
    def construct(*args, **kwargs):
        def call(*inputs):
            return layer(*(inputs + args), **kwargs)
        return call
    return construct

__all__ = [
    'DummyContextMgr',
    'make_layer'
]
