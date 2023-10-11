
from typing import List
import regex as re
import numpy as np
import pandas as pd    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from mllibs.tokenisers import custpunkttokeniser

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
    
    ldf = pd.DataFrame({'data':lst_data,
                        'tag':lst_tags})
    
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

