import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from xswem.utils import assert_layers_built
from xswem.exceptions import UnexpectedEmbeddingSizeException, WordMissingFromVocabMapException

_EMBEDDINGS_INITIALIZER_LOWER_BOUND = -0.01
_EMBEDDINGS_INITIALIZER_UPPER_BOUND = 0.01


class XSWEM(tf.keras.Model):

    def __init__(self, embedding_size, output_activation, vocab_map, output_map, mask_zero=False, dropout_rate=None,
                 output_regularizer=None, embedding_weights_map=None, adapt_embeddings=None, freeze_embeddings=None,
                 **kwargs):
        """ Model class for XSWEM.

            Parameters
            ----------
            embedding_size : int
                Number of components in each embedding vector.
            output_activation : tf.keras.activations.Activation
                Activation function to use in the output layer.
            vocab_map : dict
                Map of int to str. Describes the word corresponding to each possible int in the input to the model. If
                mask_zero, then should have all keys in the range 1 to len(vocab_map)+1, otherwise should have all keys
                in the range 0 to len(vocab_map).
            output_map : dict
                Map of int to str. Describes each output unit in English.
            mask_zero : bool
                Whether or not the input value 0 is a special "padding" value that should be masked out.
            dropout_rate : float
                Dropout rate to apply to the output of the embedding layer. Default of None, meaning no dropout.
            output_regularizer : tf.keras.regularizers.Regularizer
                Regularizer for the output layer. Default of None, meaning no regularization.
            embedding_weights_map : dict
                Map of str to np.array. Describes the weights to initialize each word embedding with. Words described in
                vocab_map but not in this map will have their weights initialized randomly using a uniform distribution
                with range -0.01 to 0.01 as described in the original paper. Default of None, meaning all weights are
                initialized randomly using the procedure described.
            adapt_embeddings : bool
                Whether to apply a Dense layer to the output of the embedding layer. The dense layer has embedding_size
                units and uses a relu activation function. This corresponds to method ii described in chapter 4 of the
                original paper. Default of False, meaning no dense layer.
            freeze_embeddings : bool
                Whether to freeze the embedding weights. Default of False, meaning do not freeze them.
        """
        super(XSWEM, self).__init__(**kwargs)
        self._embedding_size = embedding_size
        self._output_activation = output_activation
        self._vocab_map = vocab_map
        self._output_map = output_map
        self._mask_zero = mask_zero
        self._dropout_rate = dropout_rate
        self._output_regularizer = output_regularizer
        self._adapt_embeddings = adapt_embeddings
        self._embedding_weights_map = embedding_weights_map
        self._freeze_embeddings = freeze_embeddings if freeze_embeddings is not None else False
        self._kwargs = kwargs
        self._verify_vocab_map_valid()
        embeddings_initializer = tf.keras.initializers.RandomUniform(_EMBEDDINGS_INITIALIZER_LOWER_BOUND,
                                                                     _EMBEDDINGS_INITIALIZER_UPPER_BOUND)
        embedding_weights = self._prepare_word_embeddings() if embedding_weights_map else None
        embedding_input_dim = len(self._vocab_map) + 1 if self._mask_zero else len(self._vocab_map)
        self.embedding_layer = tf.keras.layers.Embedding(embedding_input_dim, self._embedding_size,
                                                         mask_zero=self._mask_zero,
                                                         embeddings_initializer=embeddings_initializer,
                                                         weights=embedding_weights,
                                                         name="Embedding",
                                                         trainable=not self._freeze_embeddings)
        if self._dropout_rate:
            self.embedding_dropout_layer = tf.keras.layers.Dropout(self._dropout_rate, name="EmbeddingDropout")
        if self._adapt_embeddings:
            self.embedding_dense_layer = tf.keras.layers.Dense(self._embedding_size, activation="relu",
                                                               name="EmbeddingDense")
        self.max_pool_layer = tf.keras.layers.GlobalMaxPool1D(name="MaxPool")
        self.output_layer = tf.keras.layers.Dense(len(self._output_map), activation=self._output_activation,
                                                  kernel_regularizer=self._output_regularizer, name="Output")

    def _verify_vocab_map_valid(self):
        """ Checks that the vocab map is in the format expected """
        vocab_map_offset = 1 if self._mask_zero else 0
        for vocab_num in range(vocab_map_offset, len(self._vocab_map) + vocab_map_offset):
            if vocab_num not in self._vocab_map:
                raise WordMissingFromVocabMapException(vocab_num)

    def _prepare_word_embeddings(self):
        """ Builds the embedding weights to initialize the embedding layer with. If a word in vocab_map is also in
            embedding_weights_map, then the weights provided there are used. Otherwise they are initialized randomly
            using a uniform distribution with range -0.01 to 0.01 as described in the original paper.

            Returns
            -------
            list of np.array
                Weights to initialize the word embedding layer with.
        """
        embedding_weights = []
        if self._mask_zero:
            embedding_weights.append(np.zeros(self._embedding_size))
        random_count = 0
        for word in self.get_vocab_ordered_by_key():
            word_vec = self._embedding_weights_map.get(word)
            if word_vec is not None:
                word_vec_size = len(word_vec)
                if word_vec_size != self._embedding_size:
                    raise UnexpectedEmbeddingSizeException(self._embedding_size, word, word_vec_size)
            else:
                random_count += 1
                word_vec = np.random.uniform(_EMBEDDINGS_INITIALIZER_LOWER_BOUND, _EMBEDDINGS_INITIALIZER_UPPER_BOUND,
                                             size=self._embedding_size)
            embedding_weights.append(word_vec)
        if random_count != 0:
            message = "{0} words had no provided weights in embedding_weights_map so their embedding's were " \
                      "initialized randomly".format(random_count)
            tf.get_logger().warn(message)
        return [np.array(embedding_weights)]

    def call(self, inputs, training=None, mask=None):
        x = self._call_embedding_block(inputs, training=training)
        x = self.max_pool_layer(x)
        return self.output_layer(x)

    def _call_embedding_block(self, inputs, training=None):
        """ For a given input, gets the embedding weights. Applies dropout and adapts embeddings where specified. """
        x = self.embedding_layer(inputs)
        if self._dropout_rate:
            x = self.embedding_dropout_layer(x, training=training)
        if self._adapt_embeddings:
            x = self.embedding_dense_layer(x)
        return x

    def get_config(self):
        return {**{
            'embedding_size': self._embedding_size,
            'output_activation': self._output_activation,
            'vocab_map': self._vocab_map,
            'output_map': self._output_map,
            'mask_zero': self._mask_zero,
            'dropout_rate': self._dropout_rate,
            'output_regularizer': self._output_regularizer,
            'embedding_weights_map': self._embedding_weights_map,
            'adapt_embeddings': self._adapt_embeddings
        }, **self._kwargs}

    def get_vocab_ordered_by_key(self):
        """ Gets the vocabulary for the embeddings ordered by key.

            Returns
            -------
            list
                The words in the vocabulary sorted in ascending order by key.
        """
        return [self._vocab_map[vocab_num] for vocab_num in sorted(self._vocab_map.keys())]

    @assert_layers_built
    def get_embedding_weights(self, return_df=None, vocab_nums=None):
        """ Gets the embedding weights for the model.

            Parameters
            ----------
            return_df : bool
                Whether to return the results as a pandas DataFrame. Default of True. If false then results are returned
                as a numpy array.
            vocab_nums : iterable
                List of ints describing which words to get the embedding weights of. If None gets the embeddings for
                all words in vocab_map.

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
        if vocab_nums is None:
            vocab_nums = sorted(list(self._vocab_map.keys()))
        embedding_weights = self._call_embedding_block(np.array([vocab_nums]), training=False).numpy()[0]
        if return_df:
            vocab = [self._vocab_map[vocab_num] for vocab_num in vocab_nums]
            return pd.DataFrame(embedding_weights, index=vocab)
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

    def _embedding_components_top_n_words(self, n, vocab_nums=None):
        embedding_weights = self.get_embedding_weights(vocab_nums=vocab_nums)
        explained_components = {**{"Word Rank": range(1, n + 1)},
                                **{column_name: embedding_weights.nlargest(n, columns=column_name).index
                                   for column_name in embedding_weights.columns}}
        return pd.DataFrame.from_dict(explained_components).set_index("Word Rank")

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
        return self._embedding_components_top_n_words(n)

    def local_explain_word_shortlist(self, pre_processed_input_sentence, by_index=None):
        """ Gets the embeddings of the words in the input_sentence and finds the word corresponding to the maximum value
            for each component (effectively the argmax of each component). This is equivalent to the output of the max
            pooling layer, except instead of the output being the maximum value for each component we have the word that
            corresponds to the maximum value. As no words outside this list contribute to the output of the max-pooling
            layer, no words outside this list contribute to the prediction of the model for the input sentence. Thus,
            these words are a shortlist of words that the model has learnt are most important to make a prediction given
            the input sentence.

            Parameters
            ----------
            pre_processed_input_sentence : np.array
                The input sentence we want a local explanation for. This should be pre-processed in the same way as the
                input to the model during training.
            by_index : bool
                Determines whether the output of the function is returned as a pd.DataFrame or np.array. Default of
                false.

            Returns
            -------
            pd.DataFrame or np.array
                If by_index is true, returns the result as a pd.DataFrame, describing the argmax word for each
                component. If by_index is false, returns the results as a np.array, listing the unique words that are
                argmax for at least one component.
        """
        unique_input_sentence = np.unique(pre_processed_input_sentence)
        word_shortlist = self._embedding_components_top_n_words(n=1, vocab_nums=unique_input_sentence)
        if not by_index:
            word_shortlist = np.sort(np.unique(word_shortlist.values))
        return word_shortlist
