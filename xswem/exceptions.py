_DEFAULT_UNBUILT_LAYERS_EXCEPTION_MESSAGE = "Some layers in the model are not built. Train the model before " \
                                            "calling this method."
_DEFAULT_UNEXPECTED_EMBEDDING_SIZE_EXCEPTION_MESSAGE = "Expected embeddings to be of size {0}, but provided " \
                                                       "embedding weights for word '{1}' is of size {2}."
_DEFAULT_WORD_MISSING_FROM_VOCAB_MAP_EXCEPTION_MESSAGE = "Expected vocab_map to contain key {0}, but it does not. If " \
                                                         "mask_zero, then vocab_map should have all keys in the " \
                                                         "range 1 to len(vocab_map)+1, otherwise should have all " \
                                                         "keys in the range 0 to len(vocab_map)."


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


class UnexpectedEmbeddingSizeException(XSWEMException):
    """ Raised when embedding weights provided for a word have more components than expected """

    def __init__(self, expected_size, unexpected_size_word, unexpected_size, message=None):
        message = (message or _DEFAULT_UNEXPECTED_EMBEDDING_SIZE_EXCEPTION_MESSAGE).format(expected_size,
                                                                                           unexpected_size_word,
                                                                                           unexpected_size)
        super(UnexpectedEmbeddingSizeException, self).__init__(message)


class WordMissingFromVocabMapException(XSWEMException):
    """ Raised when a word is missing from the vocab map """

    def __init__(self, missing_word, message=None):
        message = (message or _DEFAULT_WORD_MISSING_FROM_VOCAB_MAP_EXCEPTION_MESSAGE).format(missing_word)
        super(WordMissingFromVocabMapException, self).__init__(message)
