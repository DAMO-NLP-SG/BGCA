import json
import logging
import re
import copy

import editdistance
import numpy as np

from constants import *

logger = logging.getLogger(__name__)

sentiment_word_list = ['positive', 'negative', 'neutral']

def extract_spans_extraction(task, seq, io_format):
    if task == "uabsa":
        if io_format == "extraction":
            return extract_uabsa_from_extration(seq)
        elif io_format == "extraction-universal":
            return extract_uabsa_from_extraction_universal(seq, io_format)
        else:
            raise NotImplementedError
    elif task == "ate":
        if io_format == "extraction":
            return extract_ate_from_extraction(seq)
        elif io_format == "extraction-universal":
            return extract_ate_from_extraction_universal(seq)
        else:
            raise NotImplementedError
    elif task == "aste":
        if io_format == "extraction":
            return extract_aste_from_extraction(seq)
        elif io_format == "extraction-universal":
            return extract_aste_from_extraction_universal(seq)
        else:
            raise NotImplementedError
    elif task == "aope":
        if io_format == "extraction":
            return extract_aope_from_extraction(seq)
        elif io_format == "extraction-universal":
            return extract_aope_from_extraction_universal(seq)
        else:
            raise NotImplementedError


def extract_ate_from_extraction(seq):
    aps = re.findall("(\()(.*?)(?=\);?|$)", seq)
    pairs = []
    for ap in aps:
        special_token, aspect = ap[0].strip(), ap[1].strip()
        aspect = aspect.strip()
        pairs.append((aspect))
    return pairs


def extract_ate_from_extraction_universal(seq):

    aps = re.findall("(<aspect>)(.*?)(?=<aspect>|$)", seq)
    pairs = []
    for ap in aps:
        special_token, aspect = ap[0].strip(), ap[1].strip()
        aspect = aspect.strip()
        pairs.append((aspect))
    return pairs


def extract_aste_from_extraction_universal(seq):

    aps = re.findall("(<pos>|<neg>|<neu>)(.+?)<opinion>(.+?)(?=<pos>|<neg>|<neu>|$)", seq)
    pairs = []
    for ap in aps:
        senti, aspect, opinion = ap[0].strip(), ap[1].strip(), ap[2].strip()
        senti = SPECIAL_TO_TAG[senti]
        pairs.append((aspect, opinion, senti))
    return pairs


def extract_aste_from_extraction(seq):
    aps = re.findall("\((.+?),(.+?),(.+?)\);?", seq)
    pairs = []
    for ap in aps:
        aspect, opinion, senti = ap[0].strip(), ap[1].strip(), ap[2].strip()
        pairs.append((aspect, opinion, senti))
    return pairs


def extract_aope_from_extraction_universal(seq):

    aps = re.findall("(<aspect>)(.+?)<opinion>(.+?)(?=<aspect>|$)", seq)
    pairs = []
    for ap in aps:
        tag, aspect, opinion = ap[0].strip(), ap[1].strip(), ap[2].strip()
        pairs.append((aspect, opinion))
    return pairs


def extract_aope_from_extraction(seq):
    aps = re.findall("\((.+?),(.+?)\);?", seq)
    pairs = []
    for ap in aps:
        aspect, opinion = ap[0].strip(), ap[1].strip()
        pairs.append((aspect, opinion))
    return pairs


def extract_uabsa_from_extration(seq):
    aps = re.findall("\((.+?),(.+?)\);?", seq)
    pairs = []
    for ap in aps:
        aspect, opinion = ap[0].strip(), ap[1].strip()
        pairs.append((aspect, opinion))
    return pairs


def extract_uabsa_from_extraction_universal(seq, io_format, keep_special=False):
    """
        extraction-universal:  <pos> apple <pos> orange <neg> banana //   [none]
    """
    """
    matches:
    [('manager ', '<neg>'), (' drinks ', '<pos>'), (' appetizers ', '<pos>')]
    """
    # aps = re.findall("(<pos>|<neg>|<neu>)(.*?)(<\/pos>|<\/neg>|<\/neu>)", seq)
    if io_format == "extraction-universal":
        aps = re.findall("(<pos>|<neg>|<neu>)(.*?)(?=<pos>|<neg>|<neu>|$)", seq)
    pairs = []
    for ap in aps:
        if io_format in ["extraction-universal"]:
            special_token, aspect = ap[0].strip(), ap[1].strip()
        aspect = aspect.strip()
        if keep_special:
            senti_tag = special_token
        else:
            senti_tag = SPECIAL_TO_TAG[special_token]
        pairs.append((aspect, senti_tag))
    return pairs


def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    new_words = []
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def fix_preds_uabsa(all_pairs, sents):

    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AT not in the original sentence
                if pair[0] not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                if pair[1] not in TAG_WORD_LIST:
                    new_sentiment = recover_terms_with_editdistance(pair[1], TAG_WORD_LIST)
                else:
                    new_sentiment = pair[1]

                new_pairs.append((new_at, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_ate(all_pairs, sents):

    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AT not in the original sentence
                if pair not in  ' '.join(sents[i]):
                    # notice here pair alone is an aspect, no need pair[0]
                    new_at = recover_terms_with_editdistance(pair, sents[i])
                else:
                    new_at = pair

                new_pairs.append((new_at))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_aope(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #print(pair)
                # AT not in the original sentence
                if pair[0] not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # OT not in the original sentence
                ots = pair[1].split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)

                new_pairs.append((new_at, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_aste(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #two formats have different orders
                p0, p1, p2 = pair
                # for annotation-type
                if p1 in sentiment_word_list:
                    at, ott, ac = p0, p2, p1
                    io_format = 'annotation'
                # for extraction type
                elif p2 in sentiment_word_list:
                    at, ott, ac = p0, p1, p2
                    io_format = 'extraction'

                #print(pair)
                # AT not in the original sentence
                if at not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(at, sents[i])
                else:
                    new_at = at

                if ac not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(ac, sentiment_word_list)
                else:
                    new_sentiment = ac

                # OT not in the original sentence
                ots = ott.split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)
                if io_format == 'extraction':
                    new_pairs.append((new_at, new_ot, new_sentiment))
                else:
                    new_pairs.append((new_at, new_sentiment, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_pred_with_editdistance(all_predictions, sents, task):
    if task == "uabsa":
        fixed_preds = fix_preds_uabsa(all_predictions, sents)
    elif task == "ate":
        fixed_preds = fix_preds_ate(all_predictions, sents)
    elif task == "aste":
        fixed_preds = fix_preds_aste(all_predictions, sents)
    elif task == "aope":
        fixed_preds = fix_preds_aope(all_predictions, sents)
    else:
        logger.info("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds


def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    n_tp, n_gold, n_pred = 0, 0, 0
    gold_pt = copy.deepcopy(gold_pt)

    for i in range(len(pred_pt)):
        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                # to prevent generate same correct answer and get recall larger than 1
                gold_pt[i].remove(t)
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sents, paradigm, task, verbose=False):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        if "extraction" in paradigm:
            gold_list = extract_spans_extraction(task, gold_seqs[i], paradigm)
            pred_list = extract_spans_extraction(task, pred_seqs[i], paradigm)

            all_labels.append(gold_list)
            all_predictions.append(pred_list)

    raw_scores = compute_f1_scores(all_predictions, all_labels)
    # fix the issues due to generation
    all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task)
    fixed_scores = compute_f1_scores(all_predictions_fixed, all_labels)

    if verbose:
        for i in range(3):
            logger.info(f"Gold: {gold_seqs[i]}")
            logger.info(f"Gold list: {all_labels[i]}")
            logger.info(f"Pred: {pred_seqs[i]}")
            logger.info(f"Pred list: {all_predictions[i]}")
        logger.info("Results of raw output")
        logger.info(str(raw_scores))
        logger.info("Results of fixed output")
        logger.info(str(fixed_scores))

    return raw_scores, fixed_scores, all_labels, all_predictions, all_predictions_fixed


def avg_n_seeds_by_pair(output_dir, dirs, decode_txt, n_runs):
    score_avg_dict = {}
    score_type_list = ["raw_scores", "fixed_scores"]
    metric_list = ["precision", "recall", "f1"]
    pairs = []

    # collect value
    for dir_ in dirs:
        for score_type in score_type_list:
            if score_type not in score_avg_dict:
                score_avg_dict[score_type] = {}
            pair = dir_.split('/')[-1]
            src, tgt = pair.split('-')
            if pair not in score_avg_dict[score_type]:
                score_avg_dict[score_type][pair] = {}
            score_dict_i = json.load(open(f"{dir_}/score/test_{decode_txt}_score.json","r"))

            for metric in metric_list:
                if metric not in score_avg_dict[score_type][pair]:
                    score_avg_dict[score_type][pair][metric] = []
                score_avg_dict[score_type][pair][metric].append(score_dict_i[score_type][tgt][metric])

    # get all value
    for score_type in score_type_list:
        all_mat_dict = {k: [] for k in metric_list}
        for pair in score_avg_dict[score_type]:
            for metric in metric_list:
                f1_list_by_seed = score_avg_dict[score_type][pair][metric]
                all_mat_dict[metric].append(f1_list_by_seed)
        for metric in metric_list:
            if "all" not in score_avg_dict[score_type]:
                score_avg_dict[score_type]["all"] = {}
            score_avg_dict[score_type]["all"][metric] = np.mean(all_mat_dict[metric], axis=0)

    # avg value
    for score_type in score_type_list:
        for pair in score_avg_dict[score_type]:
            for metric in metric_list:
                mean = np.mean(score_avg_dict[score_type][pair][metric])
                std = np.std(score_avg_dict[score_type][pair][metric])
                score_avg_dict[score_type][pair][metric] = (mean, std)

    # visual result
    for score_type in score_type_list:
        logger.info('@'*100)
        logger.info(f"Avged {n_runs} runs {score_type}")
        logger.info('\t'.join(list(score_avg_dict[score_type].keys())))
        f1_list = [i["f1"][0] for i in list(score_avg_dict[score_type].values())]
        logger.info('\t'.join([f"{i*100:.2f}" for i in f1_list]))
        f1_std_list = [i["f1"][1] for i in list(score_avg_dict[score_type].values())]
        logger.info('\t'.join([f"{i*100:.2f}" for i in f1_std_list]))

    json.dump(score_avg_dict, open(output_dir+f"/score_{decode_txt}_avg.json", "w"), indent=2)