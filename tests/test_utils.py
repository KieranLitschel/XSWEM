import unittest
from unittest.mock import Mock
from xswem.utils import assert_layers_built
from xswem.exceptions import UnbuiltLayersException


class TestUtils(unittest.TestCase):

    def test_assert_layers_built(self):
        output = "pass"

        class Model(object):
            def __init__(self, layers):
                self.layers = layers

            @assert_layers_built
            def get_output(self):
                return output

        mock_layer_built = Mock()
        mock_layer_built.built = True
        mock_layer_unbuilt = Mock()
        mock_layer_unbuilt.built = False

        built_model = Model([mock_layer_built, mock_layer_built])
        self.assertEqual(built_model.get_output(), output)

        unbuilt_model = Model([mock_layer_built, mock_layer_unbuilt])
        with self.assertRaises(UnbuiltLayersException):
            unbuilt_model.get_output()


if __name__ == '__main__':
    unittest.main()
