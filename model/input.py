import csv
import itertools
import numpy as np
import nltk
from logger.log import Logger

LOGGER = Logger(__name__)
VOCAB_SIZE = 8000
UNK = "_UNK_"
START_TOKEN = "_START_"
END_TOKEN = "_END_"


def tokenize(corpus_path,
             sen_tokenizer=nltk.sent_tokenize,
             word_tokenizer=nltk.word_tokenize):
    """

    Parameters
    ----------
    corpus_path
    sen_tokenizer
    word_tokenizer

    Returns
    -------

    """
    LOGGER.info("Reading corpus file %s" % corpus_path)
    with open(corpus_path, 'r') as f:
        if corpus_path.endswith('.csv'):
            reader = csv.reader(f, skipinitialspace=True)
            sentences = itertools.chain(*[sen_tokenizer(x[0]) for x in reader])
            sentences = ["%s %s %s" % (START_TOKEN, x, END_TOKEN) for x in sentences]
            LOGGER.info("Parsed %d sentences." % (len(sentences)))
            words_list = [word_tokenizer(sent) for sent in sentences]
            return words_list
        if corpus_path.endswith('.txt'):
            sentences = itertools.chain(*[sen_tokenizer(x[0]) for x in f])
            sentences = ["%s %s %s" % (START_TOKEN, x, END_TOKEN) for x in sentences]
            LOGGER.info("Parsed %d sentences." % (len(sentences)))
            words_list = [word_tokenizer(sent) for sent in sentences]
            return words_list


def input_generator(corpus_path, path):
    """

    Parameters
    ----------
    corpus_path
    path

    Returns
    -------

    """
    pass


def corpus_digitize(sentences):
    """

    Parameters
    ----------
    sentences

    Returns
    -------

    """
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*sentences))
    LOGGER.info("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(VOCAB_SIZE - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(UNK)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    LOGGER.info("Using vocabulary size %d." % VOCAB_SIZE)
    LOGGER.info("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(sentences):
        words_list[i] = [w if w in word_to_index else UNK for w in sent]

    # Create the training data
    x = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in sentences])
    y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in sentences])
    return x, y


if __name__ == '__main__':
    corpus_path = '../data/reddit-comments-2015-08.csv'
    words_list = tokenize(corpus_path)
    x, y = corpus_digitize(words_list)
    for d in x:
        print(d)
    # print(y)

