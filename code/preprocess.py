import os
import re
import logging

from constants import *

logger = logging.getLogger(__name__)

def ot2bieos_absa(absa_tag_sequence):
    """
    ot2bieos function for end-to-end aspect-based sentiment analysis task
    """
    n_tags = len(absa_tag_sequence)
    #new_ts_sequence = []
    new_absa_sequence = []
    prev_pos = '$$$'

    for i in range(n_tags):
        cur_absa_tag = absa_tag_sequence[i]
        if cur_absa_tag == 'O' or cur_absa_tag == 'EQ':
            # when meet the EQ tag, regard it as O
            new_absa_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_absa_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_absa_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_absa_tag = absa_tag_sequence[i + 1]
                    if next_absa_tag == 'O':
                        new_absa_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_absa_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_absa_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_absa_tag = absa_tag_sequence[i + 1]
                    if next_absa_tag == 'O':
                        new_absa_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_absa_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_absa_sequence


def bieos2generation(sents, labels):
    final_sents = []

    for si, s in enumerate(sents):
        pairs = []
        aspect_idx = []
        for wi, w in enumerate(s):
            tag = labels[si][wi]
            if tag == "O":
                aspect_idx = []
                continue

            label, polarity = labels[si][wi].split('-')
            if label in ["B", "I"]:
                aspect_idx.append(wi)
            elif label in ["E", "S"]:
                aspect_idx.append(wi)
                aspect_tuple = (aspect_idx, polarity)
                pairs.append(aspect_tuple)
                aspect_idx = []

        final_s = ' '.join(s) + "#"*4 + str(pairs)
        final_sents.append(final_s)

    return final_sents


def read_generation_uabsa(file_path):
    # ["I love apple .####[([0, "POS"])]"]
    sents, labels = read_by_bieos(file_path)
    final_sents = bieos2generation(sents, labels)
    return final_sents


def read_by_bieos(file_path):
    sents, labels  = [], []
    with open(file_path, 'r', encoding='UTF-8') as fp:
        words, tags = [], []
        for line in fp:
            word_part, label_part = line.strip().split("####")
            # I=O love=O apple=T-POS
            tokens = label_part.split(" ")
            # remove some period "."
            tokens = [t for t in tokens if "=" in t]
            # sometimes there are multiple =, such as ==O
            words = ["".join(i.split("=")[:-1]) for i in tokens]
            tags = [i.split("=")[-1] for i in tokens]

            tags = ot2bieos_absa(tags)
            sents.append(words)
            labels.append(tags)
            words, tags = [], []
    return sents, labels


def write_generation(file_paths, split="train", data_dir=None, do_write=True, nrows=None, mode='aste'):

    final_sents = []
    count_dict = {}
    for path in file_paths:

        sentences, domain = None, None
        if mode == 'aste':
            sentences = open(path, "r").readlines()[:nrows]
            if split in ["dev", "test"]:
                domain = path.split('/')[-2]
        elif mode == "uabsa":  # 'uabsa'
            sentences = read_generation_uabsa(path)[:nrows]
            if split in ["dev", "test"]:
                domain = re.search(f"(\w+)[_-]{split}\.txt", path).group(1)

        count_dict[domain] = len(sentences)
        final_sents.extend(sentences)
        logger.info(f"{split} {path}: {len(sentences)}")

    logger.info(f"{split} total: {len(final_sents)}")
    logger.info(f"{split} count dict: {count_dict}")

    if do_write:
        output_dir = data_dir

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_path = f'{output_dir}/{split}.txt'
        with open (output_path, 'w') as f:
            for s in final_sents:
                f.write (f'{s.strip()}\n')
            logger.info(f"{output_path} is written.")

    return final_sents, count_dict


def preprocess(dataset_dir=None, data_dir=None, source=None, targets=None, do_write=True, nrows=None, unlabel=False, mode='uabsa'):

    if mode == 'uabsa':
        train_file_format = "{i}_train.txt"
        dev_file_format = f"{source}_dev.txt"
        test_file_format = "{i}_test.txt"
    elif mode == 'aste':
        train_file_format = "{i}/train.txt"
        dev_file_format = f"{source}/dev.txt"
        test_file_format = "{i}/test.txt"
    else:
        raise ValueError("Invalid mode. Choose either 'uabsa' or 'aste'.")

    if unlabel:
        # Process unlabeled data
        train_paths = [os.path.join(dataset_dir, train_file_format.format(i=i)) for i in targets]
        train_sents, _ = write_generation(train_paths, "target-unlabel", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)
        logger.info(f"Cross domain unlabel train_paths: {train_paths}")
        return train_sents

    else:
        # Process labeled data
        train_paths = [os.path.join(dataset_dir, train_file_format.format(i=i)) for i in [source]]
        dev_paths = [os.path.join(dataset_dir, dev_file_format)]
        test_paths = [os.path.join(dataset_dir, test_file_format.format(i=i)) for i in targets]

        logger.info(f"train_paths: {train_paths}")
        logger.info(f"dev_paths: {dev_paths}")
        logger.info(f"test_paths: {test_paths}")

        # Read and preprocess the data using write_generation
        _, _ = write_generation(train_paths, "train", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)
        _, dev_count_dict = write_generation(dev_paths, "dev", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)
        _, test_count_dict = write_generation(test_paths, "test", data_dir=data_dir, do_write=do_write, nrows=nrows, mode=mode)

        return dev_count_dict, test_count_dict


def prepare_raw_data(args):

    def process_task(preprocessor, mode, dataset_dir=None, data_gene=False, pseudo=False):
        eval_count_dict, test_count_dict = preprocessor(
            dataset_dir=dataset_dir,
            data_dir=args.data_dir,
            source=args.source_domain,
            targets=args.target_domain,
            nrows=args.nrows,
            mode=mode
        )
        if data_gene or pseudo:
            preprocessor(
                dataset_dir=dataset_dir,
                data_dir=args.data_dir,
                source=args.source_domain,
                targets=args.target_domain,
                nrows=args.nrows,
                unlabel=True,
                mode=mode
            )
        return eval_count_dict, test_count_dict

    # ate and uabsa share same dataset
    task_map = {
        "uabsa": ('uabsa',  "../data/uabsa/cross_domain"),
        "ate": ('uabsa',  "../data/uabsa/cross_domain"),
        "aste": ('aste', "../data/aste/cross_domain"),
        "aope": ('aste', "../data/aope/cross_domain")
    }

    if args.task in task_map and args.dataset == "cross_domain":
        mode, dataset_dir = task_map[args.task]
        return process_task(preprocess, mode, dataset_dir, args.data_gene, args.pseudo)
    else:
        raise NotImplementedError
