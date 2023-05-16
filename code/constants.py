T5_ORI_LEN = 32100

TAG_TO_WORD = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
TAG_WORD_LIST = ['positive', 'negative', 'neutral']

# annotation-bracket
TAG_TO_BRACKET = {"POS": ("((", "))"), "NEG": ("[[", "]]"), "NEU": ("{{", "}}")}
BRACKET_TO_TAG = {"{{": "neutral", "[[": "negative", "((": "positive"}

# annotation-special
NONE_TOKEN = "[none]"
AND_TOKEN = "[and]"
ASPECT_TOKEN = "<aspect>"
OPINION_TOKEN = "<opinion>"
EMPTY_TOKEN = "<empty>"
SEP_TOKEN = "<sep>"
TAG_TO_SPECIAL = {"POS": ("<pos>", "</pos>"), "NEG": ("<neg>", "</neg>"), "NEU": ("<neu>", "</neu>")}
SPECIAL_TO_TAG = {"<pos>": "positive", "<neg>": "negative", "<neu>": "neutral"}

TARGET_TEST_COUNT_DICT = {
    "laptop": 800,
    "rest": 2158,
    "device": 1279,
    "service": 747,
}
# please follw L, R, D, S order for evaluation purpose, since we will combine test files into one
UABSA_TRANSFER_PAIRS = {
    "laptop": ["rest", "service"],
    "rest": ["laptop", "device", "service"],
    "device": ["rest", "service"],
    "service": ["laptop", "rest", "device"],
}
ASTE_TRANSFER_PAIRS = {
    "rest14": ["laptop14"],
    "rest15": ["laptop14"],
    "rest16": ["laptop14"],
    "laptop14": ["rest14", "rest15", "rest16"],
}
AOPE_TRANSFER_PAIRS = ASTE_TRANSFER_PAIRS

STOP_WORDS = ['about', 'itself', 'so', 'further', 'against', "don't", 'shouldn', 'to', 'didn', 'hers', 'over', 'haven', "it's", 'of', 'have', 'm', 'but', "you've", 'which', 'd', 'most', 'nor', "haven't", "wasn't", 'yourself', 'with', 'am', 'do', 'than', "that'll", "isn't", 'or', "shan't", 'then', 'while', 'did', 'off', 'under', "mustn't", "won't", 'again', 'you', 'its', 'these', 'some', 'he', 'after', 'doesn', 'into', 't', 'more', 'whom', 'his', 'from', 'a', 'at', 'during', 'when', "she's", "aren't", 'was', 'same', 'myself', 'my', 'has', 'aren', 'by', 'before', "needn't", 'yourselves', 'such', 'she', 'is', 'needn', 'here', 'too', 'ourselves', "didn't", 'both', 'i', 'theirs', 'weren', 'be', 'their', 'were', 'because', 'should', "should've", "couldn't", 'will', 'isn', 'all', 'and', 'through', 'won', "weren't", 'y', 'they', 'for', 'until', 'him', 's', 'now', 'those', 'up', 'had', 'that', 'ma', 'couldn', 'been', 'why', 'below', 'own', 'doing', "you'll", 'very', 'above', "shouldn't", 'where', 've', 'if', 'are', 'how', 'wasn', 'it', 'what', 'as', 'hadn', 'hasn', "you'd", "wouldn't", 'don', 'few', 'other', 're', 'ain', "hadn't", "doesn't", 'himself', 'shan', 'the', 'not', 'mustn', 'does', "hasn't", 'll', 'your', 'yours', 'herself', 'in', 'wouldn', 'themselves', 'who', 'there', 'ours', 'out', 'mightn', 'me', 'them', 'once', "mightn't", 'we', 'her', 'this', 'being', 'any', 'can', 'o', 'no', 'having', "you're", 'our', 'on', 'between', 'down', 'only', 'just', 'each', 'an']