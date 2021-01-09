import tensorflow as tf
import numpy as np
import pandas as pd
from xswem.model import XSWEM
from xswem.exceptions import UnexpectedEmbeddingSizeException
from unittest.mock import patch, Mock


class TestXSWEM(tf.test.TestCase):
    def setUp(self):
        self.vocab_map = {1: "UNK", 2: "hello", 3: "world"}
        self.output_map = {0: "is_hello_world"}
        self.model = XSWEM(2, 'sigmoid', self.vocab_map, self.output_map)
        self.embedding_weights = [np.array([[1, -1],
                                            [2, -2],
                                            [3, -3]], dtype=np.float32)]
        self.embedding_dense_weights = [np.array([[-0.1, 0.1],
                                                  [0.1, -0.1]], dtype=np.float32),
                                        np.array([0.5, -0.1], dtype=np.float32)]
        self.output_weights = [np.array([[2],
                                         [4]], dtype=np.float32),
                               np.array([5], dtype=np.float32)]
        self.set_up_model(self.model)
        self.test_sentence = np.array([[1, 2]], dtype=np.int)
        # output of max pool should be the vector (3,-2), multiplying by the coefficients of the output layer and adding
        # the biases we get the output is 2*3 + 4*-2 + 5 = 6 - 8 + 5 = 3, the activation function is sigmoid, so the
        # output of the model should be 1/(1+e^(-3)) = 0.95257 (to 5 d.p.)
        self.expected_test_sentence_prediction = 0.95257

    @staticmethod
    def build_model(model):
        model.compile("sgd", "binary_crossentropy")
        model.build([None, 2])

    def set_up_model(self, model):
        self.build_model(model)
        model.embedding_layer.set_weights(self.embedding_weights)
        if getattr(model, "embedding_dense_layer", None):
            model.embedding_dense_layer.set_weights(self.embedding_dense_weights)
        model.output_layer.set_weights(self.output_weights)

    def get_embedding_weights_map(self):
        embedding_weights_map = {}
        for i in range(len(self.vocab_map)):
            word = self.vocab_map[i + 1]
            weights = self.embedding_weights[0][i]
            embedding_weights_map[word] = weights
        return embedding_weights_map

    def test_get_config(self):
        expected_config = {'embedding_size': 2,
                           'output_activation': 'sigmoid',
                           'vocab_map': {1: 'UNK', 2: 'hello', 3: 'world'},
                           'output_map': {0: 'is_hello_world'},
                           'mask_zero': False,
                           'dropout_rate': None,
                           'output_regularizer': None,
                           'embedding_weights_map': None,
                           'adapt_embeddings': None}
        self.assertEqual(self.model.get_config(), expected_config)

    @staticmethod
    def get_layer_types(model):
        return [type(layer) for layer in model.layers]

    def test_model_architecture(self):
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(self.get_layer_types(self.model), expected_layer_types)

    def test_call(self):
        test_sentence_prediction = self.model.call(self.test_sentence, training=False).numpy()[0][0]
        self.assertAlmostEqual(test_sentence_prediction, self.expected_test_sentence_prediction, places=5)

    def test_dropout(self):
        # mock the dropout call so that if we're training with dropout we always dropout the same components
        def mock_dropout_call(*args, **kwargs):
            training = kwargs['training']
            if training:
                return tf.convert_to_tensor([[[2, 0],
                                              [0, -3]]], dtype=tf.float32)
            else:
                return tf.convert_to_tensor([[[2, -2],
                                              [3, -3]]], dtype=tf.float32)

        with patch('tensorflow.keras.layers.Dropout.__call__', mock_dropout_call):

            model = XSWEM(2, 'sigmoid', self.vocab_map, self.output_map, dropout_rate=0.5)
            self.set_up_model(model)

            # model architecture
            expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.Dropout,
                                    tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
            self.assertListEqual(self.get_layer_types(model), expected_layer_types)

            # call test time
            test_sentence_prediction = model.call(self.test_sentence, training=False).numpy()[0][0]
            self.assertAlmostEqual(test_sentence_prediction, self.expected_test_sentence_prediction, places=5)

            # call train time
            dropout_test_sentence_prediction = model.call(self.test_sentence, training=True).numpy()[0][0]
            # we dropout some units in the embeddings, so the output of the max pool layer is (2,0), so the output of
            # the model is 2*2 + 4*0 + 5 = 9 prior to the activation, so the output after the activation should be
            # 1/(1+e^(-9)) = 0.99988 (to 5 d.p.)
            expected_dropout_test_sentence_prediction = 0.99988
            self.assertAlmostEqual(dropout_test_sentence_prediction, expected_dropout_test_sentence_prediction,
                                   places=5)

    def test_get_vocab_ordered_by_key(self):
        self.assertEqual(self.model.get_vocab_ordered_by_key(), ["UNK", "hello", "world"])

    def test_get_embedding_weights(self):
        pd_embedding_weights = self.model.get_embedding_weights()
        expected_pd_embedding_weights = pd.DataFrame(self.embedding_weights[0], columns=pd.Index([0, 1]),
                                                     index=pd.Index(["UNK", "hello", "world"]))
        self.assertIsInstance(pd_embedding_weights, pd.DataFrame)
        pd.testing.assert_frame_equal(pd_embedding_weights, expected_pd_embedding_weights)
        np_embedding_weights = self.model.get_embedding_weights(return_df=False)
        self.assertIsInstance(np_embedding_weights, np.ndarray)
        np.testing.assert_array_equal(np_embedding_weights, self.embedding_weights[0])

    def get_prepared_word_embeddings(self, embedding_weights_map):
        model = XSWEM(2, 'sigmoid', self.vocab_map, self.output_map, embedding_weights_map=embedding_weights_map)
        self.build_model(model)
        embedding_weights = model.embedding_layer.get_weights()
        self.assertLen(embedding_weights, 1)
        return embedding_weights[0]

    def test_prepare_word_embeddings(self):
        embedding_weights_map = self.get_embedding_weights_map()
        expected_embedding_weights = self.embedding_weights[0]
        RANDOM_WEIGHTS = np.array([5, 5])
        with patch('tensorflow.get_logger', new_callable=Mock()) as mock_logger:
            with patch('numpy.random.uniform', lambda *args, **kwargs: RANDOM_WEIGHTS):
                # test with all words in weights map
                embedding_weights = self.get_prepared_word_embeddings(embedding_weights_map)
                np.testing.assert_array_equal(embedding_weights, expected_embedding_weights)
                mock_logger().warn.assert_not_called()
                # test the random initialization by deleting a word embedding from the map
                del embedding_weights_map["UNK"]
                embedding_weights = self.get_prepared_word_embeddings(embedding_weights_map)
                expected_embedding_weights[0] = RANDOM_WEIGHTS
                np.testing.assert_array_equal(embedding_weights, expected_embedding_weights)
                mock_logger().warn.assert_called_once_with(
                    '1 words had no provided weights in embedding_weights_map so their embedding\'s were initialized '
                    'randomly'
                )
                # test that exception thrown when a word embedding has a different length than specified in the
                # constructor
                with self.assertRaises(UnexpectedEmbeddingSizeException):
                    embedding_weights_map["UNK"] = np.array([1])
                    XSWEM(2, 'sigmoid', self.vocab_map, self.output_map, embedding_weights_map=embedding_weights_map)

    def test_adapt_embeddings(self):
        model = XSWEM(2, 'sigmoid', self.vocab_map, self.output_map, adapt_embeddings=True)
        self.set_up_model(model)
        # architecture
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.Dense,
                                tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(self.get_layer_types(model), expected_layer_types)
        # call
        test_sentence_prediction = model.call(self.test_sentence, training=True).numpy()[0][0]
        # the word embeddings for our test sentence are (2,-2) and (3,-3) respectively. we adapt them using a dense
        # layer. the new values become (-0.1*x+0.1*y+0.5,0.1*x-0.1*y-0.1) for each vector. so our adapted embeddings are
        # (-0.2-0.2+0.5,0.2+0.2-0.1) = (0.1,0.3), and (-0.3-0.3+0.5,0.3+0.3-0.1)=(-0.1,0.5). We apply a relu activation
        # so they become (0.1,0.3) and (0,0.5) respectively. We max pool these vectors so we are left with the vector
        # (0.1,0.5). Applying our output layer the output of the network is 1/(1+e^(-(0.1*2+0.5*4+5)))=0.99925 to 5 d.p.
        expected_test_sentence_prediction = 0.99925
        self.assertAlmostEqual(test_sentence_prediction, expected_test_sentence_prediction, places=5)

    def test_dropout_and_adapt_embeddings(self):
        model = XSWEM(2, 'sigmoid', self.vocab_map, self.output_map, adapt_embeddings=True, dropout_rate=0.5)
        self.set_up_model(model)
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.Dropout, tf.keras.layers.Dense,
                                tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(self.get_layer_types(model), expected_layer_types)

    def test_global_plot_embedding_histogram(self):
        with patch('matplotlib.pyplot.show', new_callable=Mock) as mock_show:
            with patch('seaborn.histplot', new_callable=Mock) as mock_histplot:
                self.model.global_plot_embedding_histogram()
                expected_data = self.embedding_weights[0].flatten()
                mock_histplot.assert_called_once()
                np.testing.assert_array_equal(mock_histplot.call_args[0][0], expected_data)
                mock_histplot.return_value.set_title.assert_called_once_with("Histogram for Learned Word Embeddings")
                mock_histplot.return_value.set_xlabel.assert_called_once_with("Embedding Component Value")
                mock_histplot.return_value.set_ylabel.assert_called_once_with("Frequency")
                mock_show.assert_called_once()

    def test_global_explain_embedding_components(self):
        explained_components = self.model.global_explain_embedding_components(2)
        expected_explained_components = pd.DataFrame(np.array([["world", "UNK"], ["hello", "hello"]]),
                                                     index=pd.Index([1, 2], name="Word Rank"),
                                                     columns=pd.Index([0, 1], dtype=object))
        self.assertIsInstance(explained_components, pd.DataFrame)
        pd.testing.assert_frame_equal(explained_components, expected_explained_components)


if __name__ == '__main__':
    tf.test.main()
