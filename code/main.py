import json
import logging
import os
import sys

from pytorch_lightning import seed_everything
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer)

from constants import *
from data_utils import (read_line_examples_from_file, get_dataset)
from eval_utils import avg_n_seeds_by_pair, compute_scores
from preprocess import prepare_raw_data
from run_utils import (aux_training, train, infer)
from model_utils import prepare_tag_tokens, prepare_constrained_tokens, init_tag
from setup import init_args, prepare_seeds, prepare_pairs

logger = logging.getLogger()


def prepare_logger(args):
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    formatter = logFormatter = logging.Formatter(fmt='[%(asctime)s - %(name)s:%(lineno)d]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_file = os.path.join(args.seed_dir, "run.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.handlers = [console_handler, file_handler]


def evaluate(args, tokenizer, dataset, model, paradigm, task, sents, split, is_constrained, eval_set_count_dict=None, silent=False):
    """
    Compute scores given the predictions and gold labels
    """

    decode_txt = "constrained" if is_constrained else "greedy"
    logger.info(f"Eval set count dict: {eval_set_count_dict}")

    inputs, outputs, targets = infer(
        args, dataset, model, tokenizer, name=f"{split}_{task}",
        is_constrained=is_constrained, constrained_vocab=prepare_constrained_tokens(tokenizer, task, paradigm),
    )

    start_idx = 0
    score_dict = {
        "raw_scores": {k: 0 for k in eval_set_count_dict.keys()},
        "fixed_scores": {k: 0 for k in eval_set_count_dict.keys()},
    }
    pred_dict = {
        "labels": {k: 0 for k in eval_set_count_dict.keys()},
        "preds": {k: 0 for k in eval_set_count_dict.keys()},
        "preds_fixed": {k: 0 for k in eval_set_count_dict.keys()},
    }

    verbose = True if split == "dev" else False
    for l, num in eval_set_count_dict.items():
        raw_score, fixed_score, label, pred, pred_fixed = compute_scores(
                                                outputs[start_idx: start_idx+num],
                                                targets[start_idx: start_idx+num],
                                                sents[start_idx: start_idx+num],
                                                paradigm, task, verbose)
        start_idx += num

        score_dict["raw_scores"][l] = raw_score
        score_dict["fixed_scores"][l] = fixed_score
        pred_dict["labels"][l] = label
        pred_dict["preds"][l] = pred
        pred_dict["preds_fixed"][l] = pred_fixed

    """
        score_dict = {
            "raw_scores": {
                "en": {"precision": 0, "recall": 0, "f1": 0},
                "all": {"precision": 0, "recall": 0, "f1": 0},
            }
            "fixed_score": {
            }
        }
    """

    if not silent:
        for score_type in ["raw_scores", "fixed_scores"]:

            logger.info('='*100)
            logger.info(score_type)
            logger.info('\t'.join(list(score_dict[score_type].keys())))
            f1_list = [i["f1"] for i in list(score_dict[score_type].values())]
            logger.info('\t'.join([f"{i*100:.2f}" for i in f1_list]))

    with open(os.path.join(args.score_dir, f"{split}_{decode_txt}_errors.txt"), "w") as f:
        counter = 0
        for lang in pred_dict['labels'].keys():
            for i in range(len(pred_dict['labels'][lang])):
                label_i = pred_dict['labels'][lang][i]
                pred_i = pred_dict['preds'][lang][i]
                pred_fixed_i = pred_dict['preds_fixed'][lang][i]
                test_i = ' '.join(sents[counter])
                if label_i != pred_i:
                    f.write(f"{test_i} === {label_i} === {pred_i} === {pred_fixed_i} \n")
                counter += 1

    # return  {k: score_dict["fixed_scores"]["all w/t en"][k] for k in ["precision", "recall", "f1"]}
    return score_dict, pred_dict


def main(args):

    prepare_logger(args)
    start_info = "="*30+f"NEW EXP: {args.task.upper()} on {args.dataset} with seed {args.seed}"+"="*30
    logger.info("#"*len(start_info))
    logger.info(start_info)
    logger.info("#"*len(start_info))

    # Initialize args
    args.ori_tok_len = T5_ORI_LEN
    args.data_dir = os.path.join(args.seed_dir, "data")
    args.inference_dir = os.path.join(args.seed_dir, "inference")
    args.score_dir = os.path.join(args.seed_dir, "score")

    # Create inference and score directories if they do not exist
    for directory in [args.inference_dir, args.score_dir]:
        os.makedirs(directory, exist_ok=True)

    # Log and save args
    logger.info(args)
    json.dump(args.__dict__, open(args.seed_dir+"/args.json", 'w'), indent=2)

    seed_everything(args.seed)

    # prepare raw data, inlcudes [train.txt, test.txt, dev.txt, target-unlabel.txt] etc.
    eval_count_dict, test_count_dict = prepare_raw_data(args)

    tag_tokens = prepare_tag_tokens(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    # training process
    if args.do_train:
        # add special tokens
        special_tokens = tag_tokens
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Tokens added into embedding: {special_tokens}")
        logger.info(f"Tokenizer len: {len(tokenizer)}")

        if args.init_tag:
            init_tag(args, tokenizer, model, tag_tokens)

        # show one sample to check the sanity of the code and the expected output
        logger.info(f"Here is an example (from dev set) under `{args.paradigm}` paradigm:")
        dev_dataset = get_dataset(args, task=args.task, data_type="dev", tokenizer=tokenizer)
        for i in range(3):
            logger.info('Input : {}'.format(tokenizer.decode(dev_dataset[i]['source_ids'], skip_special_tokens=True)))
            logger.info('Output: {}'.format(tokenizer.decode(dev_dataset[i]['target_ids'], skip_special_tokens=True)))

        model.to(args.device)

        train_dataset = get_dataset(args, task=args.task, data_type="train", tokenizer=tokenizer)

        # aux training, inlcude [data_gene, pseudo]
        aux_outputs = aux_training(args, tokenizer, model, train_dataset)
        if args.data_gene or args.pseudo:
            if args.data_gene:
                train_dataset = aux_outputs["data_gene_dataset"]
            if args.pseudo:
                train_dataset = aux_outputs["pseudo_dataset"]
            # reload model
            if args.use_same_model:
                logger.info(f"Use the same model.")
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
                model.to(args.device)
                logger.info(f"Model reloaded with {args.model_name_or_path}")
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Tokens added into embedding: {special_tokens}")

        # official training
        train(args, tokenizer, model, train_dataset, task=args.task, epochs=args.num_train_epochs, lr=args.learning_rate,
        bs=args.train_batch_size, save_ckpt=True)

    if args.do_eval:

        logger.info("*"*20+" Conduct Evaluating"+"*"*20)

        all_checkpoints = []
        # retrieve all the saved checkpoints for model selection
        saved_model_dir = args.seed_dir
        if args.do_train:
            for f in os.listdir(saved_model_dir):
                file_name = os.path.join(saved_model_dir, f)
                if 'checkpoint' in file_name:
                    all_checkpoints.append(file_name)
        else:
            all_checkpoints.append(args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

        # conduct some selection (or not)
        logger.info(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        # load dev and test datasets
        test_sents, _ = read_line_examples_from_file(f"{args.data_dir}/test.txt")
        dev_sents, _ = read_line_examples_from_file(f"{args.data_dir}/dev.txt")

        dev_dataset = get_dataset(args, task=args.task, data_type="dev", tokenizer=tokenizer)
        test_dataset = get_dataset(args, task=args.task, data_type="test", tokenizer=tokenizer)

        decode_list = [False, True]
        if args.no_greedy:
            decode_list = [True]
        for is_constrained in decode_list:

            decode_txt = "constrained" if is_constrained else "greedy"
            logger.info("%"*100)
            logger.info(f"Decode by {decode_txt}")
            best_f1, best_checkpoint, best_epoch = -999999.0, None, None
            best_score_dict, best_pred_dict = None, None
            all_epochs = []

            score_dicts = { "dev": [], "test": []}

            for checkpoint in all_checkpoints:
                epoch = checkpoint.split('-')[-1][1:]
                all_epochs.append(epoch)

                # reload the model and conduct inference
                logger.info(f"Load the trained model from {checkpoint}...")
                model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
                model.to(args.device)

                score_dict_dev, pred_dict_dev = evaluate(args, tokenizer, dev_dataset, model, args.paradigm, args.task, dev_sents, "dev",
                                                        is_constrained=is_constrained, eval_set_count_dict=eval_count_dict, silent=True)
                score_dict_test, pred_dict_test = evaluate(args, tokenizer, test_dataset, model, args.paradigm, args.task, test_sents, "test",
                                                        is_constrained=is_constrained, eval_set_count_dict=test_count_dict, silent=True)

                if args.dataset == "cross_domain":
                    best_condition = score_dict_dev["raw_scores"][args.source_domain]['f1']
                if best_condition > best_f1:
                    best_f1 = best_condition
                    best_checkpoint = checkpoint
                    best_epoch = epoch
                    best_score_dict = score_dict_test
                    best_pred_dict = pred_dict_test

                score_dicts["dev"].append(score_dict_dev)
                score_dicts["test"].append(score_dict_test)

            json.dump(best_score_dict, open(f"{args.score_dir}/test_{decode_txt}_score.json", 'w'), indent=4)
            json.dump(best_pred_dict, open(f"{args.score_dir}/test_{decode_txt}_pred.json", 'w'), indent=4)

            # visualize evaluation results
            for split_i in ["dev", "test"]:
                separator = "\t"
                logger.info('='*100)
                logger.info(f"{split_i} result over from epochs {all_epochs}")
                for score_type in ["raw_scores", "fixed_scores"]:
                    logger.info(score_type)
                    langs = list(score_dicts[split_i][0][score_type].keys())
                    logger.info(f"lang{separator}"+f"{separator}".join(langs))
                    for i, epoch_i in enumerate(all_epochs):
                        if epoch_i == best_epoch:
                            print_line = f"epoch-{epoch_i}[best]{separator}"
                        else:
                            print_line = f"epoch-{epoch_i}{separator}"
                        for lang in langs:
                            for metric_i in ["precision", "recall", "f1"]:
                                print_line += "{:.2f}".format(100*score_dicts[split_i][i][score_type][lang][metric_i])+"/"
                            print_line += separator
                        logger.info(print_line)

            # print test results over last few steps
            logger.info(f"The best checkpoint is {best_checkpoint}")

        # only training's model needs to be deleted
        if args.clear_model and args.do_train:
            import shutil
            for checkpoint in all_checkpoints:
                if args.save_best and checkpoint == best_checkpoint:
                        continue
                else:
                    shutil.rmtree(checkpoint)
                    logger.info(f"{checkpoint} is removed")


def collate_seed_results(args, runed_dirs):
    decode_txt_list = ["constrained"] if args.no_greedy else ["greedy", "constrained"]
    for decode_txt in decode_txt_list:
        logger.info(f"Averaging {decode_txt}")
        if args.train_by_pair:
            avg_n_seeds_by_pair(args.output_dir, runed_dirs, decode_txt, args.n_runs)
        else:
            raise NotImplementedError


def run_multiple_seeds(args, seed_list):

    runed_dirs = []

    for i in range(args.n_runs):

        print(f"Running with seed {seed_list[i]}")

        pair_dict = prepare_pairs(args)
        for source in pair_dict:
            if args.train_by_pair:
                for target in pair_dict[source]:
                    start_info = "#"*20+f"Working on {source} --> {target}"+"$"*20
                    print("#"*len(start_info))
                    print(start_info)
                    print("#"*len(start_info))

                    output_dir_i = f"{args.output_dir}/seed-{seed_list[i]}/{source}-{target}"
                    os.makedirs(output_dir_i)
                    runed_dirs.append(output_dir_i)
                    args.seed_dir, args.seed = output_dir_i, seed_list[i]
                    args.source_domain, args.target_domain = source, [target]
                    main(args)
            else:
                raise NotImplementedError

    return runed_dirs


if __name__ == "__main__":
    args = init_args()
    seed_list = prepare_seeds(args.seed, args.n_runs)
    runed_dirs = run_multiple_seeds(args, seed_list)
    collate_seed_results(args, runed_dirs)