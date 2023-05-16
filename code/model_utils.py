
import logging

import torch

from constants import *

logger = logging.getLogger(__name__)

def prepare_constrained_tokens(tokenizer, task, paradigm):
    special_tokens = [tokenizer.eos_token] # add end token
    if task == "uabsa":
        if paradigm == "annotation":
            # note space will affect tokenization, make sure the special tokens align with processed method considering space
            special_tokens += ["[a|negative]", "[a|positive]", "[a|neutral]"]
        elif paradigm == "extraction":
            # (service, positive); (resturant, positive)
            special_tokens += ["( a , neutral ) ; ( a , negative ) ; ( a , positive ) "]
            special_tokens += ["None"]
        elif paradigm == "extraction-universal":
            special_tokens += [i[0] for i in TAG_TO_SPECIAL.values()]
            special_tokens += [NONE_TOKEN]
        else:
            raise NotImplementedError
    elif task == "ate":
        if paradigm == "annotation":
            special_tokens += ["[aspect]"]
        elif paradigm == "extraction":
            # (service, positive); (resturant, positive)
            special_tokens += ["( a ) ; ( a ) ;"]
            special_tokens += ["None"]
        elif paradigm == "extraction-universal":
            special_tokens += [ASPECT_TOKEN]
            special_tokens += [NONE_TOKEN]
        else:
            raise NotImplementedError
    elif task == "aste":
        if paradigm == "extraction-universal":
            special_tokens += [i[0] for i in TAG_TO_SPECIAL.values()]
            special_tokens += [OPINION_TOKEN]
        elif paradigm == "extraction":
            special_tokens += ["( a , a , negative ) ; ( a , a , neutral ) ; ( a , a , positive )"]
        else:
            raise NotImplementedError
    elif task == "aope":
        if paradigm == "extraction-universal":
            special_tokens += [ASPECT_TOKEN, OPINION_TOKEN, SEP_TOKEN]
        elif paradigm == "extraction":
            special_tokens += ["( a , a ) ; ( a , a )"]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return special_tokens


def prepare_tag_tokens(args):
    tag_tokens = []
    if args.task == "uabsa":
        if "extraction-universal" in args.paradigm:
            tag_tokens += [i[0] for i in TAG_TO_SPECIAL.values()]
            tag_tokens += [NONE_TOKEN]
        if args.data_gene:
            tag_tokens += [j for i in TAG_TO_SPECIAL.values() for j in i]
            tag_tokens += [ASPECT_TOKEN, OPINION_TOKEN, EMPTY_TOKEN, SEP_TOKEN]
    elif args.task == "ate":
        if args.paradigm == "extraction-universal":
            tag_tokens += [ASPECT_TOKEN, NONE_TOKEN, SEP_TOKEN]
    elif args.task == "aste":
        if "extraction-universal" in args.paradigm:
            tag_tokens += [i[0] for i in TAG_TO_SPECIAL.values()]
            tag_tokens += [OPINION_TOKEN, SEP_TOKEN]
    elif args.task == "aope":
        if "extraction-universal" in args.paradigm:
            tag_tokens += [ASPECT_TOKEN, OPINION_TOKEN, SEP_TOKEN]
    else:
        raise NotImplementedError

    tag_tokens = list(set(tag_tokens))
    logger.info(f"Tag tokens: {tag_tokens}")
    return tag_tokens


def init_tag(args, tokenizer, model, tag_tokens):
    if args.init_tag == "english":
        if args.paradigm == "extraction-universal":
            import re
            map_dict = {"pos": "positive", "neg": "negative", "neu": "neutral"}
            for tag_word in tag_tokens:
                tag_id = tokenizer.encode(tag_word, add_special_tokens=False)[0]
                init_word = re.sub("\W", "", tag_word).strip()
                # map senti
                if init_word in map_dict:
                    init_word = map_dict[init_word]
                # skip sep
                elif init_word == "sep":
                    continue
                init_id = tokenizer.encode(init_word, add_special_tokens=False)[0]
                with torch.no_grad():
                    model.shared.weight[tag_id] = model.shared.weight[init_id]
                logger.info(f"{tokenizer.decode(tag_id)} is init by {tokenizer.decode(init_id)}")
        elif args.paradigm == "extraction":
            pass
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError