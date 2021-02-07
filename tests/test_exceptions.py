import unittest
from xswem.exceptions import XSWEMException, UnbuiltLayersException, UnexpectedEmbeddingSizeException, \
    WordMissingFromVocabMapException, _DEFAULT_UNBUILT_LAYERS_EXCEPTION_MESSAGE, \
    _DEFAULT_UNEXPECTED_EMBEDDING_SIZE_EXCEPTION_MESSAGE, _DEFAULT_WORD_MISSING_FROM_VOCAB_MAP_EXCEPTION_MESSAGE


class TestExceptions(unittest.TestCase):
    def __init__(self, method_name):
        super().__init__(method_name)
        self.test_message = "foo bar"

    def test_XSWEMException(self):
        exception = XSWEMException(self.test_message)
        self.assertEqual(str(exception), self.test_message)

    def test_UnbuiltLayersException(self):
        exception = UnbuiltLayersException()
        self.assertEqual(str(exception), _DEFAULT_UNBUILT_LAYERS_EXCEPTION_MESSAGE)
        exception = UnbuiltLayersException(self.test_message)
        self.assertEqual(str(exception), self.test_message)

    def test_UnexpectedEmbeddingSizeException(self):
        test_args = (2, "foo", 1)
        exception = UnexpectedEmbeddingSizeException(*test_args)
        self.assertEqual(str(exception), _DEFAULT_UNEXPECTED_EMBEDDING_SIZE_EXCEPTION_MESSAGE.format(*test_args))
        test_message = "{0}{1}{2}"
        exception = UnexpectedEmbeddingSizeException(*test_args, message=test_message)
        self.assertEqual(str(exception), test_message.format(*test_args))

    def test_WordMissingFromVocabMapException(self):
        missing_word = 1
        exception = WordMissingFromVocabMapException(missing_word)
        self.assertEqual(str(exception), _DEFAULT_WORD_MISSING_FROM_VOCAB_MAP_EXCEPTION_MESSAGE.format(missing_word))
        test_message = "{0}"
        exception = WordMissingFromVocabMapException(missing_word, message=test_message)
        self.assertEqual(str(exception), test_message.format(missing_word))


if __name__ == '__main__':
    unittest.main()
