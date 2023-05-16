from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter
from mllibs.nlpi import nlpi
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')

class encoder(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'nlp_encoder'
        self.nlp_config = nlp_config 
        self.select = None
        self.data = None
        self.args = None
           
    # make selection  

    def sel(self,args:dict):
    
        self.select = args['pred_task']
        self.data = args['data']
        self.args = args    
        
        ''' select appropriate predicted method '''
        
        if(self.select == 'encoding_ohe'):
            self.ohe(self.data,self.args)
        elif(self.select == 'encoding_label'):
            self.le(self.data,self.args)
        elif(self.select == 'count_vectoriser'):
            self.cv(self.data,self.args)  
        elif(self.select == 'count_vectoriser_bigram'):
            self.cv_bigram(self.data,self.args)  
        elif(self.select == 'count_vectoriser_trigram'):
            self.cv_trigram(self.data,self.args)
        elif(self.select == 'tfidf_vectoriser'):
            self.tfidf(self.data,self.args)
            
    # one hot encode dataframe
            
    def ohe(self,data:pd.DataFrame,args):
        encoder = OneHotEncoder(sparse=False)
        vectors = encoder.fit_transform(data)
        df_matrix = pd.DataFrame(vectors,columns=encoder.categories_)
        display(df_matrix)
        
    # label encode dataframe

    def le(self,data:pd.Series,args):
        encoder = LabelEncoder()
        vectors = encoder.fit_transform(data)
        df_matrix = pd.DataFrame(vectors)
        nlpi.outputs['encoding_label'] = (df_matrix,encoder.classes_)
   
    # standard count vectoriser

    def cv(self,data:pd.DataFrame,args):
        
        data = data.iloc[:,0] # we know it has to be one column 
        vectoriser = CountVectorizer()
        vectors = vectoriser.fit_transform(list(data))        
        df_matrix = pd.DataFrame(vectors.todense(),
                                 columns=vectoriser.get_feature_names_out())
        display(df_matrix)
        
    def cv_bigram(self,data:pd.DataFrame,args):
        
        data = data.iloc[:,0] # we know it has to be one column 
        vectoriser = CountVectorizer(ngram_range=(1,2))
        vectors = vectoriser.fit_transform(list(data))        
        df_matrix = pd.DataFrame(vectors.todense(),
                                 columns=vectoriser.get_feature_names_out())
        display(df_matrix) 
        
    def cv_trigram(self,data:pd.DataFrame,args):
        
        data = data.iloc[:,0] # we know it has to be one column 
        vectoriser = CountVectorizer(ngram_range=(1,3))
        vectors = vectoriser.fit_transform(list(data))        
        df_matrix = pd.DataFrame(vectors.todense(),
                                 columns=vectoriser.get_feature_names_out())
        display(df_matrix) 
    
    # tfidf vectoriser
    
    def tfidf(self,data:pd.DataFrame,args):
        
        data = data.iloc[:,0] # we know it has to be one column 
        vectoriser = TfidfVectorizer()
        vectors = vectoriser.fit_transform(list(data))        
        df_matrix = pd.DataFrame(vectors.todense(),
                                 columns=vectoriser.get_feature_names_out())
        display(df_matrix)
        
        
# corpus for module
dict_nlpencode = {'encoding_ohe':['one hot encode',
                                 'one-hot-encode',
                                 'ohe',
                                 'one-hot encode',
                                 'encode with one-hot-encoding',
                                 'encoded with ohe'],
            
                 'encoding_label': ['label encode',
                                    'encode label',
                                    'label encoder'
                                    'encode target variable',
                                    'label encode taget variable',
                                    'LabelEncoder'],
                 
                 'count_vectoriser' : ['count vectorise',
                                       'count vectoriser',
                                       'make bag of words',
                                       'create bag of words',
                                       'bow vectorisation',
                                       'CountVectorizer'],
                  
                 'count_vectoriser_bigram' : ['count vectorise bigram',
                                              'make bag of words bigram',
                                              'create bag of words bigram',
                                              'bow bigram vectorisation',
                                              'CountVectorizer bigram'],
                  
                 'count_vectoriser_trigram' : ['count vectorise trigram',
                                              'make bag of words trigram',
                                              'create bag of words trigram',
                                              'bow trigram vectorisation',
                                              'CountVectorizer trigram'],
                                 
                  'tfidf_vectoriser': ['tfidf vectorise',
                                      'tfidf vectorisation',
                                      'tfidf',
                                      'vectorisation using tfidf',
                                      'TfidfVectorizer']
                  
                 }

# Other useful information about the task
info_nlpencode = {'encoding_ohe':{'module':'nlp_encoder',
                                  'action':'create encoding',
                                 'topic':'natural language processing',
                                  'subtopic':'create features',
                                 'input_format':'pd.DataFrame',
                                 'output_format':'pd.DataFrame',
                                 'description':'create numerical represention of feature columns containing string names'},
                  
                 'encoding_label':{'module':'nlp_encoder',
                                   'action':'create encoding',
                                   'topic':'natural language processing',
                                    'subtopic':'label encoding',
                                   'input_format':'pd.Series',
                                   'output_format':'pd.Series',
                                   'description':'create numerical presentation of target label containing string names'},
                 
                 'count_vectoriser': {'module':'nlp_encoder',
                                      'action':'create encoding',
                                      'topic':'natural language processing',
                                      'subtopic':'feature generation',
                                      'input_format':'pd.DataFrame',
                                      'output_format':'pd.DataFrame',
                                      'description':'Convert a collection of text documents to a matrix of token counts (unigrams)'},
                  

                 'count_vectoriser_bigram': {'module':'nlp_encoder',
                                             'action':'create encoding',
                                             'topic':'natural language processing',
                                              'subtopic':'feature generation',
                                             'input_format':'pd.DataFrame',
                                             'output_format':'pd.DataFrame',
                                             'description':'Convert a collection of text documents to a matrix of token counts (up to bigrams)'},
                  

                 'count_vectoriser_trigram': {'module':'nlp_encoder',
                                              'action':'create encoding',
                                              'topic':'natural language processing',
                                              'subtopic':'feature generation',
                                              'input_format':'pd.DataFrame',
                                              'output_format':'pd.DataFrame',
                                              'description':'Convert a collection of text documents to a matrix of token counts (up to trigrams)'},
                  
                 'tfidf_vectoriser': {'module':'nlp_encoder',
                                      'action':'create encoding',
                                      'topic':'natural language processing',
                                      'subtopic':'feature generation',
                                      'input_format':'pd.DataFrame',
                                      'output_format':'pd.DataFrame',
                                      'description':'Convert a collection of raw documents to a matrix of TF-IDF features'}
                 }

configure_nlpencoder = {'corpus':dict_nlpencode,'info':info_nlpencode}