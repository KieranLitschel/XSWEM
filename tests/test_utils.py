import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, mock_open
from xswem.utils import assert_layers_built, prepare_embedding_weights_map_from_glove
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

    def test_prepare_embedding_weights_map_from_glove(self):
        glove_file_path = "foo"
        vocab = ["hello", "foo", "<unk>"]
        read_data_iter = ["hello 1.1 0.2 3.3\n", "<unk> -1.1 -0.2 -3.3\n"]
        with patch('builtins.open', mock_open(read_data=''.join(read_data_iter))) as mocked_open:
            with patch('xswem.utils.tqdm', new_callable=MagicMock()) as mock_tqdm:
                # first line below is necessary for python 3.6 due to iterating being unsupported
                mocked_open.return_value.__iter__.return_value = read_data_iter
                mock_tqdm.__iter__.return_value = read_data_iter
                # test expected return with verbose off
                embedding_weights_map = prepare_embedding_weights_map_from_glove(glove_file_path, vocab)
                expected_embedding_weights_map = {
                    "hello": np.array([1.1, 0.2, 3.3], dtype=np.float32),
                    "<unk>": np.array([-1.1, -0.2, -3.3], dtype=np.float32),
                }
                self.assertCountEqual(embedding_weights_map.keys(), expected_embedding_weights_map.keys())
                np.testing.assert_array_equal(embedding_weights_map["hello"], expected_embedding_weights_map["hello"])
                np.testing.assert_array_equal(embedding_weights_map["<unk>"], expected_embedding_weights_map["<unk>"])
                mock_tqdm().__iter__.assert_not_called()
                # test verbose on
                prepare_embedding_weights_map_from_glove(glove_file_path, vocab, verbose=1)
                mock_tqdm().__iter__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
