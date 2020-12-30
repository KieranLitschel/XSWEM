_DEFAULT_UNINITIALIZED_WEIGHTS_EXCEPTION_MESSAGE = "Model weights are not initialised. Train the model before calling " \
                                                "an explain method."


class XSWEMException(Exception):
    """ Base class for exceptions in XSWEM """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class UninitializedWeightsException(XSWEMException):
    """ Raised when a method requires the model weights to be initialized but they are uninitialized. """

    def __init__(self, message=None):
        message = message or _DEFAULT_UNINITIALIZED_WEIGHTS_EXCEPTION_MESSAGE
        super(UninitializedWeightsException, self).__init__(message)
