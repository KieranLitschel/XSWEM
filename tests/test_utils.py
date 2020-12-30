import unittest
from xswem.utils import assert_layers_built
from xswem.exceptions import UnbuiltLayersException


class TestUtils(unittest.TestCase):

    def test_assert_layers_built(self):
        output = "pass"

        class Layer(object):
            def __init__(self, built):
                self.built = built

        class Model(object):
            def __init__(self, layers):
                self.layers = layers

            @assert_layers_built
            def get_output(self):
                return output

        built_model = Model([Layer(True), Layer(True)])
        self.assertEqual(built_model.get_output(), output)

        unbuilt_model = Model([Layer(True), Layer(False)])
        with self.assertRaises(UnbuiltLayersException):
            unbuilt_model.get_output()


if __name__ == '__main__':
    unittest.main()
