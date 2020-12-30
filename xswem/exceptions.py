_DEFAULT_UNBUILT_LAYERS_EXCEPTION_MESSAGE = "Some layers in the model are not built. Train the model before " \
                                            "calling this method."


class XSWEMException(Exception):
    """ Base class for exceptions in XSWEM """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class UnbuiltLayersException(XSWEMException):
    """ Raised when a method requires the model's layers to be built but they are not. """

    def __init__(self, message=None):
        message = message or _DEFAULT_UNBUILT_LAYERS_EXCEPTION_MESSAGE
        super(UnbuiltLayersException, self).__init__(message)
