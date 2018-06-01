import os
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import re
import operator
import uuid
import copy
import texttable as tt


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1

    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID/ID to item) from a dictionary.
    Items are ordered by decresaing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}

    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]

    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')

    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')

    return new_tags


def insert_singletons(x, id2singleton):
    """
    Replace word id with 0 if its frequency = 1.
    + id2singleton: a dict of words whose frequency = 1
    """
    x_array = np.array(x)
    is_singleton = np.array([True if idx in id2singleton else False for idx in x], dtype=np.bool)
    bool_mask = np.random.randint(0, 2, size=is_singleton[is_singleton].shape).astype(np.bool)
    is_singleton[is_singleton] = bool_mask
    r = np.zeros(x_array.shape)
    x_array[is_singleton] = r[is_singleton]

    return x_array


def match(p_entities, r_entities, type):
    """
    Match predicted entities with gold entities.
    """
    p_entities = [tuple(entity) for entity in p_entities if entity[-1] == type]
    r_entities = [tuple(entity) for entity in r_entities if entity[-1] == type]
    pcount = len(p_entities)
    rcount = len(r_entities)
    correct = len(list(set(p_entities) & set(r_entities)))
    return [pcount, rcount, correct]


def is_float(value):
    """
    Check in value is of type float()
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def evaluate(model, y_preds, y_reals, raw_xs):
    """
    Evaluate predictions.
    """
    parameters = model.parameters
    id_to_tag = model.id_to_tag
    assert len(y_preds) == len(y_reals)

    eval_id = str(uuid.uuid4())
    output_path = os.path.join(parameters['path_eval_result'], "eval.%s.output" % eval_id)
    scores_path = os.path.join(parameters['path_eval_result'], "eval.%s.scores" % eval_id)
    file = open(output_path, 'w')
    file_score = open(scores_path, 'w')

    print("Evaluating model with eval_id = " + eval_id)

    # Init result list[precision, recall, F] for each entity type
    elist = [v.split('-', 1)[1] for k, v in id_to_tag.items() if v.split('-', 1)[0] == 'B']
    result = [[0] * 3] * len(elist)
    #top_result = copy.copy(result)
    #low_result = copy.copy(result)

    # For layer and level evaluation
    num_layers = len(y_reals[0])
    layer_result = [[0] * 3] * num_layers
    #rest_result_p = [[0] * 3] * num_layers
    #rest_result_r = [[0] * 3] * num_layers

    for i, sentence in enumerate(raw_xs):
        file.write(' '.join(sentence) + '\n')

        p_tags = [[id_to_tag[int(y)] for y in y_pred] for y_pred in y_preds[i]]
        r_tags = [[id_to_tag[int(y)] for y in y_real] for y_real in y_reals[i]]
        p_entities = collect_entity(p_tags, -1)
        r_entities = collect_entity(r_tags, -1)

        pred_num_layers = len(y_preds[i])
        # Layer evaluation
        for layer in range(num_layers):
            if layer < pred_num_layers:
                p_layer_entities = collect_entity(p_tags[layer], layer)
            else:
                p_layer_entities = []
            r_layer_entities = collect_entity(r_tags[layer], layer)
            layer_result[layer], _, __ = eval_level('layer', r_layer_entities, p_layer_entities, layer_result[layer],
                                                    elist)

            #rest_result_p[layer], _, __ = eval_level('layer', r_entities, p_layer_entities, rest_result_p[layer], elist)
            #rest_result_r[layer], _, __ = eval_level('layer', r_layer_entities, p_entities, rest_result_r[layer], elist)

        file.write('predict|' + '|'.join(
            [','.join([str(entity[0]), str(entity[1]), entity[2]]) for entity in p_entities]) + '\n')
        file.write('gold|' + '|'.join(
            [','.join([str(entity[0]), str(entity[1]), entity[2]]) for entity in r_entities]) + '\n\n')

        result, p_entities, r_entities = eval_level('overall', r_entities, p_entities, result, elist)
        #low_result, p_entities, r_entities = eval_level('low', r_entities, p_entities, low_result, elist)
        #top_result, p_entities, r_entities = eval_level('top', r_entities, p_entities, top_result, elist)

    tab = tt.Texttable()
    # tab.set_deco((tt.Texttable.HEADER|tt.Texttable.HLINES))
    tab.set_cols_width([35, 10, 10, 10, 10, 10, 10])
    tab.set_cols_align(["l"] * 7)
    headings = ['Category', 'Precision', 'Recall', 'F-score', 'Predicts', 'Golds', 'Correct']
    tab.header(headings)
    f = measures(result, elist, tab)

    #f_top = measures(top_result, elist, tab)
    #f_low = measures(low_result, elist, tab)

    elist_layer = ['Layer {}'.format(i) for i in range(num_layers)]
    #elist_level = ['Rest Layer {}'.format(i) for i in range(num_layers)]
    _ = measures(layer_result, elist_layer, tab, 'Layer')
    #_ = measures(rest_result_p, elist_level, tab, 'Level')
    #_ = measures(rest_result_r, elist_level, tab, 'Level')

    file.close()
    file_score.write(tab.draw())
    file_score.close()
    print(tab.draw())
    os.remove(output_path)
    os.remove(scores_path)
    return round(f, 2)


def permutate_list(lst, indices, inv):
    """
    Sort lst by indices.
    """
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


def cost_matrix(dico, cost):
    """
    Reset a cost matrix for CRF to restrict illegal labels.
    Dico is the label dictionary (id_to_tag),cost matrix
    is the transition between two labels
    """
    infi_value = -10000

    for key, value in dico.items():
        tag = value.split('-')[0]
        if tag == 'O':
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I']
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'B':
            sem = value.split('-')[1]
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I' and v.split('-')[1] != sem]
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'I':
            sem = value.split('-')[1]
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I' and v.split('-')[1] != sem]
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'BOS':
            iindex = [k for (k, v) in dico.items() if v.split('-')[0] == 'I']
            for i in iindex:
                cost[key][i] = infi_value

        elif tag == 'EOS':
            cost[key] = infi_value

    return cost


def collect_entity_layer(lst):
    """
    Collect predicted tags(e.g. BIO)
    in order to get entities including nested ones
    """
    entities = []
    for i, tag in enumerate(lst):
        if tag.split('-', 1)[0] == 'B':
            entities.append([i, i + 1, tag.split('-', 1)[1]])
        elif tag.split('-', 1)[0] == 'I':
            try:
                entities[-1][1] += 1
            except:
                pass

    return entities


def collect_entity(lst, layer):
    """
    Collect predicted tags(e.g. BIO)
    in order to get entities including nested ones
    """
    entities = []
    if layer == -1:
        for itemlst in lst:
            layer_entities = collect_entity_layer(itemlst)
            entities.extend(layer_entities)
        entitylst = remove_dul(entities)
        return entitylst
    else:
        entities = collect_entity_layer(lst)
        return entities


def remove_dul(entitylst):
    """
    Remove duplicate entities in one sequence.
    """
    entitylst = [tuple(entity) for entity in entitylst]
    entitylst = set(entitylst)
    entitylst = [list(entity) for entity in entitylst]

    return entitylst


def eval_level(level, golds, preds, result, elist):
    """
    Evaluate on low/top/overall level.
    Low level: innermost entities
    Top level: outermost entities
    Overall level: overall entities.
    """
    if level == 'low':
        # Low evaluation
        low_preds, preds = low_entities(preds)
        low_golds, golds = low_entities(golds)
        result = [list(map(operator.add, result[j], match(low_preds, low_golds, etype))) for j, etype
                  in enumerate(elist)]
    elif level == 'top':
        # Top evaluation
        top_preds, preds = top_entities(preds)
        top_golds, golds = top_entities(golds)
        result = [list(map(operator.add, result[j], match(top_preds, top_golds, etype))) for j, etype
                  in enumerate(elist)]
    elif level == 'overall':
        result = [list(map(operator.add, result[j], match(preds, golds, etype))) for j, etype
                  in enumerate(elist)]
    elif level == 'layer':
        temp_result = [[0] * 3] * len(elist)
        temp_result = [list(map(operator.add, temp_result[j], match(preds, golds, etype))) for j, etype
                       in enumerate(elist)]
        temp = []
        temp.append(sum([item[0] for item in temp_result]))
        temp.append(sum([item[1] for item in temp_result]))
        temp.append(sum([item[2] for item in temp_result]))
        result = list(map(operator.add, temp, result))

    return result, preds, golds


def measures(lst, e_lst, table, mode=None):
    """
    Calculate precision, recall and F-score.
    """

    table.set_cols_dtype(['t', 'f', 'f', 'f', 'i', 'i', 'i'])
    p_count = 0
    r_count = 0
    correct = 0
    fs = []
    for j, etype in enumerate(e_lst):
        p, r, f = p_r_f(lst[j])
        fs.append(round(f, 2))
        p_count += lst[j][0]
        r_count += lst[j][1]
        correct += lst[j][-1]
        table.add_row(tuple([etype, p, r, f, lst[j][0], lst[j][1], lst[j][2]]))
    overall_p, overall_r, overall_f = p_r_f([p_count, r_count, correct])
    table.add_row(tuple(['Overall', overall_p, overall_r, overall_f,
                         p_count, r_count, correct]))

    if mode in ['Level', 'Layer']:
        return fs
    return overall_f


def p_r_f(lst):
    p_count, r_count, correct = lst[0], lst[1], lst[-1]
    if correct == 0 and r_count != 0:
        p = r = f = 0
    elif correct == 0 and r_count == 0:
        p = r = f = 100
    else:
        p = correct / p_count * 100
        r = correct / r_count * 100
        f = 2 * p * r / (p + r)

    return p, r, f


def count_nest_level(terms):
    """
    Calculate nest level of each term and
    get the max nest level.
    """
    if len(terms) == 0:
        return 1, terms

    terms = sorted(terms, key=lambda x: (-int(x[1]), int(x[0])))
    for i, term in reversed(list(enumerate(terms))):
        # nest level of flat entities is 1
        nest_level = 1
        if i == len(terms) - 1:
            terms[i].append(nest_level)
            continue

        remains = terms[i + 1:]
        levels = [nest_level]
        for item in remains:
            if int(item[0]) >= int(term[0]) and int(item[1]) <= int(term[1]):
                levels.append(nest_level + item[-1])
        terms[i].append(max(levels))

    max_nest_level = max([item[-1] for item in terms])
    return max_nest_level, terms


def top_entities(terms):
    """
    Collect top level entities.
    """
    if len(terms) == 0:
        return terms, terms
    terms = sorted(terms, key=lambda x: (-int(x[1]), int(x[0])))
    for i, term in reversed(list(enumerate(terms))):
        if i == 0:
            term.append(True)
            continue

        remains = terms[: i]
        flat = True
        for j, item in enumerate(remains):
            if int(item[0]) <= int(term[0]) and int(item[1]) >= int(term[1]):
                flat = False
                break
        terms[i].append(flat)
    top_nes = [term[0:-1] for term in terms if term[-1] == True]
    terms = [term[0:-1] for term in terms]
    return top_nes, terms


def low_entities(terms):
    """
    Get low entities.
    """
    _, terms = count_nest_level(terms)
    low_nes = [term[0:-1] for term in terms if term[-1] == 1]

    terms = [term[0:-1] for term in terms]
    return low_nes, terms
