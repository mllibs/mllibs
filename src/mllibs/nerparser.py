
from typing import List
import regex as re
import numpy as np
import pandas as pd    
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mllibs.tokenisers import custpunkttokeniser,nltk_wtokeniser
from string import punctuation

'''

PARSER FOR THE DATASET NER TAG FORMAT

'''

# Tokenisation patten
PUNCTUATION_PATTERN = r"([,\/#!$%\^&\*;:{}=\-`~()'\"’¿])"
# RE patterns for tag extraction
LABEL_PATTERN = r"\[(.*?)\]"

class Parser:
    
    # initialise, first word/id tag is O (outside)
    def __init__(self):
        self.tag_to_id = {
            "O": 0
        }
        self.id_to_tag = {
            0: "O"
        }
        
    ''' CREATE TAGS '''
        
    # input : sentence, tagged sentence
        
    def __call__(self, sentence: str, annotated: str) -> List[str]:
        
        ''' Create Dictionary of Identified Tags'''
        
        # 1. set label B or I    
        matches = re.findall(LABEL_PATTERN, annotated)
        word_to_tag = {}
        
        for match in matches:            
            if(" : " in match):
                tag, phrase = match.split(" : ")
                words = phrase.split(" ") 
                word_to_tag[words[0]] = f"B-{tag.upper()}"
                for w in words[1:]:
                    word_to_tag[w] = f"I-{tag.upper()}"
                
        ''' Tokenise Sentence & add tags to not tagged words (O)'''
                
        # 2. add token tag to main tag dictionary

        tags = []
        sentence = re.sub(PUNCTUATION_PATTERN, r" \1 ", sentence)
        
        for w in sentence.split():
            if w not in word_to_tag:
                tags.append("O")
            else:
                tags.append(word_to_tag[w])
                self.__add_tag(word_to_tag[w])
                
        return tags
    
    ''' TAG CONVERSION '''
    
    # to word2id (tag_to_id)
    # to id2word (id_to_tag)

    def __add_tag(self, tag: str):
        if tag in self.tag_to_id:
            return
        id_ = len(self.tag_to_id)
        self.tag_to_id[tag] = id_
        self.id_to_tag[id_] = tag
        
        ''' Get Tag Number ID '''
        # or just number id for token
        
    def get_id(self, tag: str):
        return self.tag_to_id[tag]
    
    ''' Get Tag Token from Number ID'''
    # given id get its token
    
    def get_label(self, id_: int):
        return self.get_tag_label(id_)

'''

Create NER 

'''

def ner_model(parser,df):

    # parse our NER tag data & tokenise our text
    lst_data = []; lst_tags = []
    for ii,row in df.iterrows():
        sentence = re.sub(PUNCTUATION_PATTERN, r" \1 ", row['question'])
        lst_data.extend(sentence.split())
        lst_tags.extend(parser(row["question"], row["annotated"]))
    
    ldf = pd.DataFrame({'data':lst_data,'tag':lst_tags})
    
    ''' 
    
    Vectorisation 
    
    '''
        
    # define encoder
    encoder = CountVectorizer(tokenizer=custpunkttokeniser)
    
    # fit the encoder on our corpus
    X = encoder.fit_transform(lst_data)
    y = np.array(lst_tags)
    
    ''' 
    
    Modeling 
    
    '''
    
    # try our different models
    # model_confirm = LogisticRegression()
    model_confirm = RandomForestClassifier()
    
    # train model
    model_confirm.fit(X,y)
    y_pred = model_confirm.predict(X)
    # print(f'accuracy: {round(accuracy_score(y_pred,y),3)}')
    # print(classification_report(y, y_pred))
    # print(confusion_matrix(y,y_pred))

    return model_confirm,encoder



'''

Full Variant of Feature DictExtraction

'''

def extract_token_features_full(tokens: list):
    token_features = []
    
    for i, token in enumerate(tokens):
        features = {
            'token': token,
            'is_first_token': i == 0,
            'is_last_token': i == len(tokens) - 1,
            'is_capitalized': token[0].isupper(),
            'is_alphanumeric': token.isalnum(),
            'is_punctuation': token in punctuation
        }

        if i < len(tokens) - 1:
            next_token = tokens[i+1]
            features['next_token_p1'] = next_token
            features['is_next_first_token_p1'] = i + 1 == 0
            features['is_next_last_token_p1'] = i + 1 == len(tokens) - 2
            features['is_next_numeric_p1'] = next_token.isdigit()
            features['is_next_alphanumeric_p1'] = next_token.isalnum()
            features['is_next_punctuation_p1'] = next_token in punctuation
        else:
            features['next_token_p1'] = None
            features['is_next_first_token_p1'] = False
            features['is_next_last_token_p1'] = False
            features['is_next_numeric_p1'] = False
            features['is_next_alphanumeric_p1'] = False
            features['is_next_punctuation_p1'] = False
        
        if i > 1:
            prev_token = tokens[i-1]
            features['prev_token_m1'] = prev_token
            features['is_prev_first_token_m1'] = i - 1 == 0
            features['is_prev_last_token_m1'] = i - 1 == len(tokens) - 2
            features['is_prev_numeric_m1'] = prev_token.isdigit()
            features['is_prev_alphanumeric_m1'] = prev_token.isalnum()
            features['is_prev_punctuation_m1'] = prev_token in punctuation
        else:
            features['prev_token_m1'] = None
            features['is_prev_first_token_m1'] = False
            features['is_prev_last_token_m1'] = False
            features['is_prev_numeric_m1'] = False
            features['is_prev_alphanumeric_m1'] = False
            features['is_prev_punctuation_1'] = False

        if i < len(tokens) - 2:
            next_token = tokens[i+2]
            features['next_token_p2'] = next_token
            features['is_next_first_token_p2'] = i + 1 == 0
            features['is_next_last_token_p2'] = i + 1 == len(tokens) - 2
            features['is_next_numeric_p2'] = next_token.isdigit()
            features['is_next_alphanumeric_p2'] = next_token.isalnum()
            features['is_next_punctuation_p2'] = next_token in punctuation
        else:
            features['next_token_p2'] = None
            features['is_next_first_token_p2'] = False
            features['is_next_last_token_p2'] = False
            features['is_next_numeric_p2'] = False
            features['is_next_alphanumeric_p2'] = False
            features['is_next_punctuation_p2'] = False

        if i > 2:
            prev_token = tokens[i-2]
            features['prev_token_m2'] = prev_token
            features['is_prev_first_token_m2'] = i - 1 == 0
            features['is_prev_last_token_m2'] = i - 1 == len(tokens) - 2
            features['is_prev_numeric_m2'] = prev_token.isdigit()
            features['is_prev_alphanumeric_m2'] = prev_token.isalnum()
            features['is_prev_punctuation_m2'] = prev_token in punctuation
        else:
            features['prev_token_m2'] = None
            features['is_prev_first_token_m2'] = False
            features['is_prev_last_token_m2'] = False
            features['is_prev_numeric_m2'] = False
            features['is_prev_alphanumeric_m2'] = False
            features['is_prev_punctuation_m2'] = False

        token_features.append(features)
        
    return token_features

'''

Smaller Variant

'''

def extract_token_features_small(tokens:list):
    
    token_features = []
    for i, token in enumerate(tokens):
        features = {
            'token': token,
            'is_first_token': i == 0,
            'is_last_token': i == len(tokens) - 1,
            'is_capitalized': token[0].isupper(),
            'is_all_caps': token.isupper(),
            'is_numeric': token.isdigit(),
            'is_alphanumeric': token.isalnum(),
            'is_punctuation': token in punctuation
        }
        token_features.append(features)
        
    return token_features


'''
############################################################

tf-idf transformer approach to NER

        need to tokenise first; use whitespace tokeniser
        so its the same as the dicttransformer

############################################################
'''

def tfidf(tokens:list,vectoriser=None):
    
    # tokenise by whitespaces (include dots)
    if(vectoriser == None):
        vectoriser = TfidfVectorizer(tokenizer=lambda x: nltk_wtokeniser(x),token_pattern=None)
        X = vectoriser.fit_transform(tokens)
        return X,vectoriser
    else:
        X = vectoriser.transform(tokens)
        return X,None

'''
############################################################

                dicttransformers approach to NER

                  created for each token in list

############################################################
'''

def dicttransformer(tokens:list,vectoriser=None):

    # Extract token-level features for each token
    token_features = extract_token_features_full(tokens)
    
    # Vectorize the token features
    if(vectoriser == None):
        vectoriser = DictVectorizer()
        X = vectoriser.fit_transform(token_features) # also sparse
        return X,vectoriser
    else:
        X = vectoriser.transform(token_features) # also sparse
        return X,None
        
    return X,vectoriser

'''

Merge and Predict

'''

# merge tf-idf & dict features & train model
def merger_train(X1,X2,y):

    # convert to non-sparse 
    X_vect1 = pd.DataFrame(np.asarray(X1.todense()))
    X_vect2 = pd.DataFrame(np.asarray(X2.todense()))
    data = pd.concat([X_vect1,X_vect2],axis=1)
    data.fillna(0.0,inplace=True)
    data = data.values
    
    # # try:
    # #     # Load the model from the file
    # #     model = CatBoostClassifier(silent=True)
    # #     model.load_model('ner_catboost.bin')
    # #     print('[note] using cached ner catboost model')
    # # except:
    # #     model = CatBoostClassifier(silent=True)
    # #     model.fit(data,y)
    # #     model.save_model('ner_catboost.bin')

    model = GradientBoostingClassifier()
    model.fit(data,y)
    return data,model

# merge tf-idf & dict features & train model

def merger(X1,X2):

    # convert to non-sparse 
    X_vect1 = pd.DataFrame(np.asarray(X1.todense()))
    X_vect2 = pd.DataFrame(np.asarray(X2.todense()))
    data = pd.concat([X_vect1,X_vect2],axis=1)
    data.fillna(0.0,inplace=True)
    data = data.values # convert to numpy

    return data

# predict & measure metric

def predict_label(X,tokens,labels,model):
    y_pred = model.predict(X)
    accuracy = accuracy_score(labels, y_pred)
    print(f'accuracy: {round(accuracy_score(y_pred,labels),3)}')
    print(classification_report(labels, y_pred))
    print(confusion_matrix(labels,y_pred))
    # display(pd.DataFrame({'y':tokens,
    #                       'yp':list(itertools.chain(*y_pred))}).T)

# predict & measure metric (show misspredictions)

def predict_label_missprediction(X, tokens, labels, model):
    y_pred = model.predict(X)
    mispredictions = []
    
    for i in range(len(y_pred)):
        if y_pred[i] != labels[i]:
            mispredictions.append((tokens[i], labels[i], y_pred[i]))
    
    accuracy = accuracy_score(labels, y_pred)
    print(f'accuracy: {round(accuracy_score(y_pred, labels), 3)}')
    print(classification_report(labels, y_pred))
    print(confusion_matrix(labels, y_pred))
    return mispredictions
