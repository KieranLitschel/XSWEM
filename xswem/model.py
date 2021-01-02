import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from xswem.utils import assert_layers_built


class XSWEM(tf.keras.Model):

    def __init__(self, embedding_size, output_activation, vocab_map, output_map, mask_zero=False, dropout_rate=None,
                 output_regularizer=None, **kwargs):
        """ Model class for XSWEM.

            Parameters
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

    def get_vocab_ordered_by_key(self):
        """ Gets the vocabulary for the embeddings ordered by key.

            Returns
            -------
            list
                The words in the vocabulary sorted in ascending order by key.
        """
        return [self._vocab_map[i] for i in sorted(self._vocab_map.keys())]

    @assert_layers_built
    def get_embedding_weights(self, return_df=None):
        """ Gets the embedding weights for the model.

            Parameters
            ----------
            return_df : bool
                Whether to return the results as a pandas DataFrame. Default of True. If false then results are returned
                as a numpy array.

            Returns
            -------
            pd.DataFrame or np.array
                If return_df, then returns a DataFrame of the embedding weights. Each row in the DataFrame corresponds
                to a word in the vocabulary. The index describes the corresponding word for each row. The column
                describes which component the weights are for, and are labelled 0 to number_of_components-1. If
                return_df is False, then returns a numpy array of the embeddings weights in the same format as the
                DataFrame.
        """
        return_df = return_df if return_df is not None else True
        embedding_weights = self.embedding_layer.weights[0].numpy()
        if return_df:
            return pd.DataFrame(embedding_weights, index=self.get_vocab_ordered_by_key())
        else:
            return embedding_weights

    def global_plot_embedding_histogram(self):
        """ Plots a histogram of the flattened embedding weights. This graph is equivalent to figure 1 in the original
            paper.
        """
        embedding_weights_flat = self.get_embedding_weights(return_df=False).flatten()
        ax = sns.histplot(embedding_weights_flat)
        ax.set_title("Histogram for Learned Word Embeddings")
        ax.set_xlabel("Embedding Component Value")
        ax.set_ylabel("Frequency")
        plt.show()

    def global_explain_embedding_components(self, n=None):
        """ Gets the words corresponding to the n largest values for each component of the word embedding. These can be
            analysed to determine the meaning of each component (column) as discussed in section 4.1.1 of the original
            paper.

            Parameters
            ----------
            n : int
                Number of words to return for each component. Default of 5 as proposed in the original paper.

            Returns
            -------
            pd.DataFrame
                Returns a DataFrame similar to table 3 in the original paper. The columns describe which component the
                values are for, and are labelled 0 to number_of_components-1. The values in each column are the words
                corresponding to the n largest values for the described component, sorted from the largest to smallest
                values.
        """
        n = n or 5
        embedding_weights = self.get_embedding_weights()
        explained_components = {**{"Word Rank": range(1, n + 1)},
                                **{column_name: embedding_weights.nlargest(n, columns=column_name).index
                                   for column_name in embedding_weights.columns}}
        return pd.DataFrame.from_dict(explained_components).set_index("Word Rank")
