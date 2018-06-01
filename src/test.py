import os
import sys
os.environ['CHAINER_SEED'] = '0'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle

import chainer.functions as F
from chainer import iterators
from chainer import cuda
from chainer import serializers

from src.model.layered_model import Model, Evaluator, Updater
from src.model.loader import load_sentences, update_tag_scheme, parse_config
from src.model.loader import prepare_dataset
from src.model.utils import evaluate


def predict(data_iter, model, mode):
    """
    Iterate data with well - trained model
    """
    for batch in data_iter:
        raw_words = [x['str_words'] for x in batch]
        words = [model.xp.array(x['words']).astype('i') for x in batch]
        chars = [model.xp.array(y).astype('i') for x in batch for y in x['chars']]
        tags = model.xp.vstack([model.xp.array(x['tags']).astype('i') for x in batch])

        # Init index to keep track of words
        index_start = model.xp.arange(F.hstack(words).shape[0])
        index_end = index_start + 1
        index = model.xp.column_stack((index_start, index_end))

        # Maximum number of hidden layers = maximum nested level + 1
        max_depth = len(batch[0]['tags'][0])
        sentence_len = np.array([x.shape[0] for x in words])
        section = np.cumsum(sentence_len[:-1])
        predicts_depths = model.xp.empty((0, int(model.xp.sum(sentence_len)))).astype('i')

        for depth in range(max_depth):
            next, index, extend_predicts, words, chars = model.predict(chars, words, tags[:, depth], index, mode)
            predicts_depths = model.xp.vstack((predicts_depths, extend_predicts))
            if not next:
                break

        predicts_depths = model.xp.split(predicts_depths, section, axis=1)
        ts_depths = model.xp.split(model.xp.transpose(tags), section, axis=1)
        yield ts_depths, predicts_depths, raw_words


def load_mappings(mappings_path):
    """
    Load mappings of:
      + id_to_word
      + id_to_tag
      + id_to_char
    """
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
        id_to_word = mappings['id_to_word']
        id_to_char = mappings['id_to_char']
        id_to_tag = mappings['id_to_tag']

    return id_to_word, id_to_char, id_to_tag


def main(config_path):
    args = parse_config(config_path)

    # Load sentences
    test_sentences = load_sentences(args["path_test"], args["replace_digit"])

    # Update tagging scheme (IOB/IOBES)
    update_tag_scheme(test_sentences, args["tag_scheme"])

    # Load mappings from disk
    id_to_word, id_to_char, id_to_tag = load_mappings(args["mappings_path"])
    word_to_id = {v: k for k, v in id_to_word.items()}
    char_to_id = {v: k for k, v in id_to_char.items()}
    tag_to_id  = {v: k for k, v in id_to_tag.items()}

    # Index data
    test_data = prepare_dataset(test_sentences, word_to_id, char_to_id, tag_to_id, None, args["lowercase"])
    test_iter = iterators.SerialIterator(test_data, args["batch_size"], repeat=False, shuffle=False)

    model = Model(len(word_to_id), len(char_to_id), len(tag_to_id), args)

    serializers.load_npz(args['path_model'], model)

    model.id_to_tag = id_to_tag
    model.parameters = args

    device = args['gpus']
    if device['main'] >= 0:
        cuda.get_device_from_id(device['main']).use()
        model.to_gpu()

    pred_tags = []
    gold_tags = []
    words = []

    # Collect predictions
    for ts, ys, xs in predict(test_iter, model, args['mode']):
        gold_tags.extend(ts)
        pred_tags.extend(ys)
        words.extend(xs)

    evaluate(model, pred_tags, gold_tags, words)


if __name__ == '__main__':
    main('../src/config')
