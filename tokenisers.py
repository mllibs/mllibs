# from nltk.tokenize import word_tokenize, WhitespaceTokenizer 
import regex as re
    
def punktokeniser(text:str) -> list:
    pattern = r'\w+'
    pattern = r'\w+|,'
    words = re.findall(pattern, text)
    return words

# def nltk_wtokeniser(text):
# return WhitespaceTokenizer().tokenize(text)

# returns tokenised spans [(a,b),...]
# def nltk_wtokeniser_span(text):
#    return list(WhitespaceTokenizer().span_tokenize(text)) 

# def nltk_tokeniser(text):
#    return word_tokenize(text)

# pattern for tokenisation
PUNCTUATION_PATTERN = r"([,\/#!$%\^&\*;:{}=\-`~()'\"’¿])"

# customiser tokeniser
def custpunkttokeniser(inputs):
    PUNCTUATION_PATTERN = r"([,\/#!$%\^&\*;:{}=\-`~()'\"’¿])"
    sentence = re.sub(PUNCTUATION_PATTERN, r" \1 ", inputs)
    return sentence.split()

# create N-GRAMS from tokens
def n_grams(tokens:list,n:int):
    lst_ngrams = [' '.join(i) for i in [tokens[i:i+n] for i in range(len(tokens)-n+1)]]
    return lst_ngrams
