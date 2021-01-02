import tensorflow as tf
import numpy as np
from xswem.model import XSWEM
from unittest.mock import patch


class TestXSWEM(tf.test.TestCase):
    def setUp(self):
        self.vocab_map = {1: "UNK", 2: "hello", 3: "world"}
        self.output_map = {0: "is_hello_world"}
        self.model = XSWEM(2, 'sigmoid', self.vocab_map, self.output_map)
        self.embedding_weights = [np.array([[1, -1],
                                            [2, -2],
                                            [3, -3]], dtype=np.float32)]
        self.output_weights = [np.array([[2],
                                         [4]], dtype=np.float32),
                               np.array([5], dtype=np.float32)]
        self.set_up_model(self.model)
        self.test_sentence = np.array([[1, 2]], dtype=np.int)
        # output of max pool should be the vector (3,-2), multiplying by the coefficients of the output layer and adding
        # the biases we get the output is 2*3 + 4*-2 + 5 = 6 - 8 + 5 = 3, the activation function is sigmoid, so the
        # output of the model should be 1/(1+e^(-3)) = 0.95257 (to 5 d.p.)
        self.expected_test_sentence_prediction = 0.95257

    def set_up_model(self, model):
        model.compile("sgd", "binary_crossentropy")
        model.build([None, 2])
        model.embedding_layer.set_weights(self.embedding_weights)
        model.output_layer.set_weights(self.output_weights)

    def test_get_config(self):
        expected_config = {'embedding_size': 2,
                           'output_activation': 'sigmoid',
                           'vocab_map': {1: 'UNK', 2: 'hello', 3: 'world'},
                           'output_map': {0: 'is_hello_world'},
                           'mask_zero': False,
                           'dropout_rate': None,
                           'output_regularizer': None}
        self.assertEqual(self.model.get_config(), expected_config)

    def test_model_architecture(self):
        self.assertIsInstance(self.model.layers[0], tf.keras.layers.Embedding)
        self.assertIsInstance(self.model.layers[1], tf.keras.layers.GlobalMaxPooling1D)
        self.assertIsInstance(self.model.layers[2], tf.keras.layers.Dense)

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
            self.assertIsInstance(model.layers[0], tf.keras.layers.Embedding)
            self.assertIsInstance(model.layers[1], tf.keras.layers.Dropout)
            self.assertIsInstance(model.layers[2], tf.keras.layers.GlobalMaxPooling1D)
            self.assertIsInstance(model.layers[3], tf.keras.layers.Dense)

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


if __name__ == '__main__':
    tf.test.main()
