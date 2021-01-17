import tensorflow as tf
import numpy as np
import pandas as pd
from xswem.model import XSWEM
from xswem.exceptions import UnexpectedEmbeddingSizeException
from unittest.mock import patch, Mock
import copy

VOCAB_MAP = {1: "UNK", 2: "hello", 3: "world"}
OUTPUT_MAP = {0: "is_hello_world"}
EMBEDDING_WEIGHTS = [np.array([[1, -1],
                               [2, -2],
                               [3, -3]], dtype=np.float32)]
EMBEDDING_WEIGHTS_MAP = {
    "UNK": np.array([1, -1], dtype=np.float32),
    "hello": np.array([2, -2], dtype=np.float32),
    "world": np.array([3, -3], dtype=np.float32)
}
EMBEDDING_DENSE_WEIGHTS = [np.array([[-0.1, 0.1],
                                     [0.1, -0.1]], dtype=np.float32),
                           np.array([0.5, -0.1], dtype=np.float32)]
OUTPUT_WEIGHTS = [np.array([[2],
                            [4]], dtype=np.float32),
                  np.array([5], dtype=np.float32)]
TEST_SENTENCE = np.array([[1, 2]], dtype=np.int)
# output of max pool should be the vector (3,-2), multiplying by the coefficients of the output layer and adding
# the biases we get the output is 2*3 + 4*-2 + 5 = 6 - 8 + 5 = 3, the activation function is sigmoid, so the
# output of the model should be 1/(1+e^(-3)) = 0.95257 (to 5 d.p.)
EXPECTED_BASIC_TEST_SENTENCE_PREDICTION = 0.95257


def build_model(**kwargs):
    model = XSWEM(2, 'sigmoid', VOCAB_MAP, OUTPUT_MAP, **kwargs)
    model.compile("sgd", "binary_crossentropy")
    model.build([None, 2])
    return model


def set_up_model(**kwargs):
    model = build_model(**kwargs)
    model.embedding_layer.set_weights(EMBEDDING_WEIGHTS)
    if getattr(model, "embedding_dense_layer", None):
        model.embedding_dense_layer.set_weights(EMBEDDING_DENSE_WEIGHTS)
    model.output_layer.set_weights(OUTPUT_WEIGHTS)
    return model


def get_layer_types(model):
    return [type(layer) for layer in model.layers]


class ArchitectureIndependentTests(object):
    """ Tests which are independent of architecture """

    @staticmethod
    def test_get_embedding_weights_df(model):
        df_embedding_weights = model.get_embedding_weights()
        expected_df_embedding_weights = pd.DataFrame(EMBEDDING_WEIGHTS[0], columns=pd.Index([0, 1]),
                                                     index=pd.Index(["UNK", "hello", "world"]))
        pd.testing.assert_frame_equal(df_embedding_weights, expected_df_embedding_weights)

    @staticmethod
    def test_get_embedding_weights_np(model):
        np_embedding_weights = model.get_embedding_weights(return_df=False)
        np.testing.assert_array_equal(np_embedding_weights, EMBEDDING_WEIGHTS[0])

    @staticmethod
    def test_global_plot_embedding_histogram(model, expected_data, adapt_embeddings=None):
        with patch('matplotlib.pyplot.show', new_callable=Mock) as mock_show:
            with patch('seaborn.histplot', new_callable=Mock) as mock_histplot:
                model.global_plot_embedding_histogram(adapt_embeddings=adapt_embeddings)
                mock_histplot.assert_called_once()
                np.testing.assert_array_almost_equal(mock_histplot.call_args[0][0], expected_data)
                mock_histplot.return_value.set_title.assert_called_once_with("Histogram for Learned Word Embeddings")
                mock_histplot.return_value.set_xlabel.assert_called_once_with("Embedding Component Value")
                mock_histplot.return_value.set_ylabel.assert_called_once_with("Frequency")
                mock_show.assert_called_once()

    @staticmethod
    def test_global_explain_embedding_components(model):
        explained_components = model.global_explain_embedding_components(2)
        expected_explained_components = pd.DataFrame(np.array([["world", "UNK"], ["hello", "hello"]]),
                                                     index=pd.Index([1, 2], name="Word Rank"),
                                                     columns=pd.Index([0, 1], dtype=object))
        pd.testing.assert_frame_equal(explained_components, expected_explained_components)

    @staticmethod
    def test_frozen_embeddings(model):
        model.fit(TEST_SENTENCE, np.array([[1.0]], dtype=np.float32), epochs=1, verbose=0)
        embedding_weights = model.get_embedding_weights(return_df=False)
        np.testing.assert_array_equal(embedding_weights, EMBEDDING_WEIGHTS[0])

    @staticmethod
    def test_unfrozen_embeddings(model):
        model.fit(TEST_SENTENCE, np.array([[1.0]], dtype=np.float32), epochs=1, verbose=0)
        embedding_weights = model.get_embedding_weights(return_df=False)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(embedding_weights, EMBEDDING_WEIGHTS[0])


class TestXSWEMBasic(tf.test.TestCase):
    """ Tests most basic XSWEM architecture """

    def setUp(self):
        self.model = set_up_model()

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

    def test_model_architecture(self):
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(get_layer_types(self.model), expected_layer_types)

    def test_call(self):
        test_sentence_prediction = self.model.call(TEST_SENTENCE, training=False).numpy()[0][0]
        self.assertAlmostEqual(test_sentence_prediction, EXPECTED_BASIC_TEST_SENTENCE_PREDICTION, places=5)

    def test_get_vocab_ordered_by_key(self):
        self.assertEqual(self.model.get_vocab_ordered_by_key(), ["UNK", "hello", "world"])

    def test_get_embedding_weights(self):
        ArchitectureIndependentTests.test_get_embedding_weights_df(self.model)
        ArchitectureIndependentTests.test_get_embedding_weights_np(self.model)

    def test_global_plot_embedding_histogram(self):
        expected_data = EMBEDDING_WEIGHTS[0].flatten()
        ArchitectureIndependentTests.test_global_plot_embedding_histogram(self.model, expected_data)

    def test_global_explain_embedding_components(self):
        ArchitectureIndependentTests.test_global_explain_embedding_components(self.model)

    def test_unfrozen_embeddings(self):
        ArchitectureIndependentTests.test_unfrozen_embeddings(self.model)


class TestXSWEMDropoutEmbedding(tf.test.TestCase):
    """ Tests basic XSWEM with dropout applied to embeddings """

    def setUp(self):
        self.model = set_up_model(dropout_rate=0.5)

    def test_model_architecture(self):
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.Dropout,
                                tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(get_layer_types(self.model), expected_layer_types)

    @staticmethod
    def mock_dropout_call(*args, **kwargs):
        """ Mock the dropout call so that if we're training with dropout we always dropout the same components """
        training = kwargs['training']
        if training:
            return tf.convert_to_tensor([[[2, 0],
                                          [0, -3]]], dtype=tf.float32)
        else:
            return tf.convert_to_tensor([[[2, -2],
                                          [3, -3]]], dtype=tf.float32)

    def test_call_train(self):
        with patch('tensorflow.keras.layers.Dropout.__call__', self.mock_dropout_call):
            dropout_test_sentence_prediction = self.model.call(TEST_SENTENCE, training=True).numpy()[0][0]
            # we dropout units as defined in mock_dropout_call, so the output of the max pool layer is (2,0), so the
            # output of the model is 2*2 + 4*0 + 5 = 9 prior to the activation, so the output after the activation
            # should be 1/(1+e^(-9)) = 0.99988 (to 5 d.p.)
            expected_dropout_test_sentence_prediction = 0.99988
            self.assertAlmostEqual(dropout_test_sentence_prediction, expected_dropout_test_sentence_prediction,
                                   places=5)

    def test_call_test(self):
        with patch('tensorflow.keras.layers.Dropout.__call__', self.mock_dropout_call):
            test_sentence_prediction = self.model.call(TEST_SENTENCE, training=False).numpy()[0][0]
            self.assertAlmostEqual(test_sentence_prediction, EXPECTED_BASIC_TEST_SENTENCE_PREDICTION, places=5)


class TestXSWEMPrepareEmbeddings(tf.test.TestCase):
    """ Tests basic XSWEM with pre-trained embeddings """

    def setUp(self):
        self.embedding_weights_map = copy.deepcopy(EMBEDDING_WEIGHTS_MAP)
        self.expected_embedding_weights = copy.deepcopy(EMBEDDING_WEIGHTS)[0]
        self.random_weights = np.array([5, 5])

    def get_prepared_word_embeddings(self, embedding_weights_map):
        model = build_model(embedding_weights_map=embedding_weights_map)
        embedding_weights = model.embedding_layer.get_weights()
        self.assertLen(embedding_weights, 1)
        return embedding_weights[0]

    def test_all_words(self):
        with patch('tensorflow.get_logger', new_callable=Mock()) as mock_logger:
            with patch('numpy.random.uniform', lambda *args, **kwargs: self.random_weights):
                embedding_weights = self.get_prepared_word_embeddings(self.embedding_weights_map)
                np.testing.assert_array_equal(embedding_weights, self.expected_embedding_weights)
                mock_logger().warn.assert_not_called()

    def test_word_missing(self):
        with patch('tensorflow.get_logger', new_callable=Mock()) as mock_logger:
            with patch('numpy.random.uniform', lambda *args, **kwargs: self.random_weights):
                del self.embedding_weights_map["UNK"]
                embedding_weights = self.get_prepared_word_embeddings(self.embedding_weights_map)
                self.expected_embedding_weights[0] = self.random_weights
                np.testing.assert_array_equal(embedding_weights, self.expected_embedding_weights)
                mock_logger().warn.assert_called_once_with(
                    '1 words had no provided weights in embedding_weights_map so their embedding\'s were initialized '
                    'randomly'
                )

    def test_unexpected_embedding_size(self):
        with self.assertRaises(UnexpectedEmbeddingSizeException):
            self.embedding_weights_map["UNK"] = np.array([1])
            build_model(embedding_weights_map=self.embedding_weights_map)


class TestXSWEMAdaptFrozenEmbeddings(tf.test.TestCase):
    """ Tests basic XSWEM with adapted frozen embeddings """

    def setUp(self):
        self.model = set_up_model(adapt_embeddings=True, freeze_embeddings=True)
        # our word embeddings are (1,-1), (2,-2), and (3,-3). we adapt them using embedding_dense_layer. the new
        # values become (-0.1*x+0.1*y+0.5,0.1*x-0.1*y-0.1) for each vector. so our adapted embeddings are
        # (-0.1-0.1+0.5,0.1+0.1-0.1) = (0.3,0.1), (-0.2-0.2+0.5,0.2+0.2-0.1) = (0.1,0.3), and
        # (-0.3-0.3+0.5,0.3+0.3-0.1) = (-0.1,0.5). a relu activation is applied to them, so the final vectors are
        # (0.3,0.1), (0.1,0.3), and (0,0.5) respectively.
        self.expected_adapted_embedding_weights = np.array([[0.3, 0.1],
                                                            [0.1, 0.3],
                                                            [0.0, 0.5]], dtype=np.float32)

    def test_model_architecture(self):
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.Dense,
                                tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(get_layer_types(self.model), expected_layer_types)

    def test_call(self):
        test_sentence_prediction = self.model.call(TEST_SENTENCE, training=True).numpy()[0][0]
        # the word embeddings for our test sentence are (2,-2) and (3,-3) respectively. we adapt them using a dense
        # layer. as described above these adapted vectors are (0.1,0.3) and (0,0.5) respectively. We max pool these
        # vectors so we are left with the vector (0.1,0.5). Applying our output layer the output of the network is
        # 1/(1+e^(-(0.1*2+0.5*4+5)))=0.99925 to 5 d.p.
        expected_test_sentence_prediction = 0.99925
        self.assertAlmostEqual(test_sentence_prediction, expected_test_sentence_prediction, places=5)

    def test_get_embedding_weights(self):
        ArchitectureIndependentTests.test_get_embedding_weights_df(self.model)
        ArchitectureIndependentTests.test_get_embedding_weights_np(self.model)
        df_embedding_weights = self.model.get_embedding_weights(adapt_embeddings=True)
        expected_df_embedding_weights = pd.DataFrame(self.expected_adapted_embedding_weights, columns=pd.Index([0, 1]),
                                                     index=pd.Index(["UNK", "hello", "world"]))
        pd.testing.assert_frame_equal(df_embedding_weights, expected_df_embedding_weights)
        np_embedding_weights = self.model.get_embedding_weights(return_df=False, adapt_embeddings=True)
        np.testing.assert_array_almost_equal(np_embedding_weights, self.expected_adapted_embedding_weights)

    def test_global_plot_embedding_histogram(self):
        expected_data = EMBEDDING_WEIGHTS[0].flatten()
        ArchitectureIndependentTests.test_global_plot_embedding_histogram(self.model, expected_data)
        expected_data = self.expected_adapted_embedding_weights.flatten()
        ArchitectureIndependentTests.test_global_plot_embedding_histogram(self.model, expected_data,
                                                                          adapt_embeddings=True)

    def test_global_explain_embedding_components(self):
        ArchitectureIndependentTests.test_global_explain_embedding_components(self.model)
        explained_adapted_components = self.model.global_explain_embedding_components(2, adapt_embeddings=True)
        expected_explained_adapted_components = pd.DataFrame(np.array([["UNK", "world"], ["hello", "hello"]]),
                                                             index=pd.Index([1, 2], name="Word Rank"),
                                                             columns=pd.Index([0, 1], dtype=object))
        pd.testing.assert_frame_equal(explained_adapted_components, expected_explained_adapted_components)

    def test_frozen_embeddings(self):
        ArchitectureIndependentTests.test_frozen_embeddings(self.model)


class TestXSWEMDropoutAdaptEmbeddings(tf.test.TestCase):
    """ Tests basic XSWEM with dropout applied to embeddings and adapted embeddings """

    def setUp(self):
        self.model = set_up_model(adapt_embeddings=True, dropout_rate=0.5)

    def test_model_architecture(self):
        expected_layer_types = [tf.keras.layers.Embedding, tf.keras.layers.Dropout, tf.keras.layers.Dense,
                                tf.keras.layers.GlobalMaxPooling1D, tf.keras.layers.Dense]
        self.assertListEqual(get_layer_types(self.model), expected_layer_types)


if __name__ == '__main__':
    tf.test.main()
