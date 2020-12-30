import unittest
from xswem.exceptions import XSWEMException, UnbuiltLayersException, \
    _DEFAULT_UNBUILT_LAYERS_EXCEPTION_MESSAGE


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


if __name__ == '__main__':
    unittest.main()
