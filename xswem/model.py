import tensorflow as tf


class XSWEM(tf.keras.Model):

    def __init__(self, embedding_size, output_activation, vocab_map, output_map, mask_zero=False, dropout_rate=None,
                 output_regularizer=None, **kwargs):
        """
        Model class for XSWEM.

        Attributes
        ----------
        embedding_size : int
            Number of components in each embedding vector.
        output_activation : tf.keras.activations
            Activation function to use in the output layer.
        vocab_map : dict
            Map of int to str. Describes the word corresponding to each int in the input.
        output_map : dict
            Map of int to str. Describes each output unit in English.
        mask_zero : bool
            Whether or not the input value 0 is a special "padding" value that should be masked out.
        dropout_rate : float
            Dropout rate of the input to the output layer. Default of None, meaning no dropout.
        output_regularizer : tf.keras.regularizers.Regularizer
            Regularizer for the output layer. Default of None, meaning no regularization.
        """
        super(XSWEM, self).__init__(**kwargs)
        self._embedding_size = embedding_size
        self._output_activation = output_activation
        self._vocab_map = vocab_map
        self._output_map = output_map
        self._mask_zero = mask_zero
        self._dropout_rate = dropout_rate
        self._output_regularizer = output_regularizer
        self._kwargs = kwargs
        self.embedding_layer = tf.keras.layers.Embedding(len(self._vocab_map), self._embedding_size,
                                                         mask_zero=self._mask_zero, name="Embedding")
        if self._dropout_rate:
            self.dropout_layer = tf.keras.layers.Dropout(self._dropout_rate, name="Dropout")
        self.max_pool_layer = tf.keras.layers.GlobalMaxPool1D(name="MaxPool")
        self.output_layer = tf.keras.layers.Dense(len(self._output_map), activation=self._output_activation,
                                                  kernel_regularizer=self._output_regularizer, name="Output")

    def call(self, inputs, training=None, mask=None):
        x = self.embedding_layer(inputs)
        if self._dropout_rate:
            x = self.dropout_layer(x, training=training)
        x = self.max_pool_layer(x)
        return self.output_layer(x)

    def get_config(self):
        return {**{
            'embedding_size': self._embedding_size,
            'output_activation': self._output_activation,
            'vocab_map': self._vocab_map,
            'output_map': self._output_map,
            'mask_zero': self._mask_zero,
            'dropout_rate': self._dropout_rate,
            'output_regularizer': self._output_regularizer,
        }, **self._kwargs}
