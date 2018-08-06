import os
os.environ['CHAINER_SEED'] = '0'
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import itertools
import matplotlib as mpl
mpl.use('Agg')

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda

from src.model.layered_model import Model, Evaluator, Updater
from src.model.loader import load_sentences, update_tag_scheme, parse_config
from src.model.loader import word_mapping, char_mapping, entity_mapping, entity_tags
from src.model.loader import prepare_dataset, augment_with_pretrained
from src.model.loader import load_cost_matrix
from src.early_stopping_trigger import EarlyStoppingTrigger


def main(config_path):
    # Init args
    args = parse_config(config_path)

    # Load sentences
    train_sentences = load_sentences(args["path_train"], args["replace_digit"])
    dev_sentences = load_sentences(args["path_dev"], args["replace_digit"])

    # Update tagging scheme (IOB/IOBES)
    update_tag_scheme(train_sentences, args["tag_scheme"])
    update_tag_scheme(dev_sentences, args["tag_scheme"])

    # Create a dictionary / mapping of words
    if args['path_pre_emb']:
        dico_words_train = word_mapping(train_sentences, args["lowercase"])[0]
        dico_words, word_to_id, id_to_word, pretrained = augment_with_pretrained(
            dico_words_train.copy(),
            args['path_pre_emb'],
            list(itertools.chain.from_iterable([[w[0] for w in s] for s in dev_sentences])))
    else:
        dico_words, word_to_id, id_to_word = word_mapping(train_sentences, args["lowercase"])
        dico_words_train = dico_words

    # Create a dictionary and a mapping for words / POS tags / tags
    dico_chars, char_to_id, id_to_char = char_mapping(train_sentences + dev_sentences)
    dico_entities, entity_to_id, id_to_entity = entity_mapping(train_sentences + dev_sentences)

    # Set id of tag 'O' as 0 in order to make it easier for padding
    # Resort id_to_tag
    id_to_tag, tag_to_id = entity_tags(id_to_entity)

    if args["use_singletons"]:
        singletons = set([word_to_id[k] for k, v in dico_words_train.items() if v == 1])
    else:
        singletons = None

    # Index data
    train_data = prepare_dataset(train_sentences, word_to_id, char_to_id, tag_to_id, singletons, args["lowercase"])
    dev_data = prepare_dataset(dev_sentences, word_to_id, char_to_id, tag_to_id, None, args["lowercase"])
    print("%i / %i sentences in train / dev." % (len(train_data), len(dev_data)))

    # Init model
    model = Model(len(word_to_id), len(char_to_id), len(tag_to_id), args)

    if args['gpus']['main'] >= 0:
        cuda.get_device_from_id(args['gpus']['main']).use()
        model.to_gpu()

    print('Saving the mappings to disk...')
    model.save_mappings(id_to_word, id_to_char, id_to_tag, args)

    if args['path_pre_emb']:
        print("Loading pretrained embedding...")
        model.load_pretrained(args['path_pre_emb'])

    result_path = '../result/'

    # Init Iterators
    train_iter = chainer.iterators.SerialIterator(train_data, model.batch_size)
    dev_iter = chainer.iterators.SerialIterator(dev_data, model.batch_size, repeat=False)

    # Reset cost matrix
    id_to_tag = model.id_to_tag
    cost = model.crf.cost.data
    model.crf.cost.data = load_cost_matrix(id_to_tag, cost)

    # Init Optimizer
    optimizer = chainer.optimizers.Adam(model.lr_param)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(model.threshold))
    optimizer.add_hook(chainer.optimizer.WeightDecay(model.decay_rate))

    # Init early_stopping_trigger
    early_stopping_trigger = EarlyStoppingTrigger(args["epoch"],
                                                  key='dev/main/fscore',
                                                  eps=args["early_stopping_eps"],
                                                  early_stopping=args["early_stopping"])

    # Init Updater, Trainer and Evaluator
    updater = Updater(train_iter, optimizer, args['gpus'])
    trainer = training.Trainer(updater, stop_trigger=early_stopping_trigger, out=result_path)
    trainer.extend(Evaluator(dev_iter, optimizer.target, args['gpus']))

    # Save the best model
    trainer.extend(extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
                   trigger=training.triggers.MaxValueTrigger('dev/main/fscore'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'dev/main/loss',
         'main/accuracy', 'dev/main/accuracy',
         'elapsed_time']))

    if extensions.PlotReport.available():
        # Plot graph for loss,accuracy and fscore for each epoch
        trainer.extend(extensions.PlotReport(['main/loss', 'dev/main/loss'],
                                             x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'dev/main/accuracy'],
                                             x_key='epoch', file_name='accuracy.png'))
        trainer.extend(extensions.PlotReport(['dev/main/fscore'],
                                             x_key='epoch', file_name='fscore.png'))

    trainer.run()


if __name__ == '__main__':
    main('../src/config')
