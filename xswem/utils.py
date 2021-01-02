import functools
from xswem.exceptions import UnbuiltLayersException


def assert_layers_built(f):
    """ Wraps function f with method that checks if the model's layers are built, and if not throw an
        UnbuiltLayersException.

        Parameters
        ----------
        f : function
            Non-static function to wrap with the check method.
    """

    @functools.wraps(f)
    def decor(self, *args, **kwargs):
        if not all(layer.built for layer in self.layers):
            raise UnbuiltLayersException()
        return f(self, *args, **kwargs)

    return decor
