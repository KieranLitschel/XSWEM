import functools
import numpy as np
from xswem.exceptions import UnbuiltLayersException
from tqdm import tqdm


def assert_layers_built(f):
    """ Wraps function f with method that checks if the model's layers are built, and if not throw an
        UnbuiltLayersException.

        Parameters
        ----------
        f : function
            Non-static function to wrap with the check method.
    """

    @functools.wraps(f)
    def decor(self, *args, **kwargs):
        if not all(layer.built for layer in self.layers):
            raise UnbuiltLayersException()
        return f(self, *args, **kwargs)

    return decor


def prepare_embedding_weights_map_from_glove(glove_file_path, vocab, verbose=0):
    """ Builds a embedding_weights_map from a file in the same format as pre-trained glove word embeddings.
        You can download pre-trained Glove embeddings here https://github.com/stanfordnlp/GloVe.

        Parameters
        ----------
        glove_file_path : str
            Location of the unzipped Glove text file.
        vocab : iterable of str
            List of words to to include in the embedding_weights_map. Make sure they are in the same format as Glove
            e.g. cased/uncased. Note that Glove uses "<unk>" as the unknown word token.
        verbose : int
            Verbosity mode. 0 = silent, 1 = progress bar.

        Returns
        -------
        dict of str -> np.array
            Produces embedding_weights_map which can be used to initialize the models word embeddings. The keys of the
            dict are the words that are in the vocab and glove file, the values are the corresponding pre-trained
            weights. Note that if a word is in the vocab but not the glove file, it won't be included in this dict.
    """

    vocab = set(vocab)
    embedding_weights_map = {}
    with open(glove_file_path, "r") as file:
        if verbose:
            file = tqdm(file)
        for line in file:
            line = line.rstrip().split(" ")
            word, vector = line[0], line[1:]
            if word in vocab:
                embedding_weights_map[word] = np.array(vector, dtype=np.float32)
    return embedding_weights_map
