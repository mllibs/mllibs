from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from collections import Counter
from mllibs.nlpi import nlpi
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from sklearn.base import clone
from copy import deepcopy
import torch
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence

'''

Encoding Text Data

'''

class encoder(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'nlp_encoder'
        self.nlp_config = nlp_config 
        self.select = None
        self.data = None
        self.args = None
        
    # verbose output
        
    @staticmethod
    def verbose_set(verbose):
        print(f'set {verbose}')
          
    # set function parameter (related to activation)
            
    @staticmethod
    def sfp(args,preset,key:str):
        
        if(args[key] is not None):
            return eval(args[key])
        else:
            return preset[key] 
        
    # set general parameter
        
    @staticmethod
    def sgp(args,key:str):
        
        if(args[key] is not None):
            return eval(args[key])
        else:
            return None
           
    # make selection  

    def sel(self,args:dict):
    
        self.select = args['pred_task']
        self.subset = args['subset']
        self.data = args['data']
        self.args = args    
        
        
        ''' select appropriate predicted method '''
        
        if(self.select == 'encoding_ohe'):
            self.ohe(self.data,self.args)
        elif(self.select == 'encoding_label'):
            self.le(self.data,self.args)
        elif(self.select == 'count_vectoriser'):
            self.cv(self.data,self.args)  
        elif(self.select == 'tfidf_vectoriser'):
            self.tfidf(self.data,self.args)
        elif(self.select == 'torch_text_encode'):
            self.text_torch_encoding(self.data,self.args)
            
            
    ''' One Hot Encode DataFrame '''
            
    def ohe(self,data:pd.DataFrame,args):
           
        # if just data is specified
        if(self.subset is None):     
            df_matrix = pd.get_dummies(data)
            nlpi.memory_output.append({'data':df_matrix})
        else:      
            df_matrix = pd.get_dummies(data,columns=self.subset)
            nlpi.memory_output.append({'data':df_matrix})
                   
        
    ''' Label Encode DataFrame column '''

    def le(self,data:pd.DataFrame,args):
        
        encoder = LabelEncoder()
        data = deepcopy(data)
        
        if(self.subset is None):
        
            lencoder = clone(encoder)           
            vectors = lencoder.fit_transform(data)
            df_matrix = pd.DataFrame(vectors,columns=list(data.columns))
            nlpi.memory_output.append({'data':df_matrix,
                                      'vectoriser':lencoder})   
            
        else:
            
            lst_df = []
            for column in self.subset:    
                lencoder = clone(encoder)
                vectors = lencoder.fit_transform(data[[column]])
                df_matrix = pd.DataFrame(vectors,columns=[column])
                lst_df.append(df_matrix)
                
            # remove rows
            data.drop(self.subset,axis=1,inplace=True)
                
            # add encoded data back into data
            
            if(len(lst_df) > 1):
                grouped_labels = pd.concat(lst_df,axis=1)
                add_label = pd.concat([data,grouped_labels],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                          'vectoriser':lencoder})      
            else:     
                add_label = pd.concat([data,lst_df[0]],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                      'vectoriser':lencoder})      
                
   
    ''' CountVectoriser '''

    def cv(self,data:pd.DataFrame,args):
                    
        # preset value dictionary
        pre = {'ngram_range':(1,1),'min_df':1,'max_df':1.0}
        data = deepcopy(data)
        
        vectoriser = CountVectorizer(ngram_range=self.sfp(args,pre,'ngram_range'),
                                     min_df=self.sfp(args,pre,'min_df'),
                                     max_df=self.sfp(args,pre,'max_df'),
                                     tokenizer=args['tokeniser'])
        
        if(self.subset is None):
        
            data = data.iloc[:,0] # we know it has to be one column 
            vectors = vectoriser.fit_transform(list(data))        
            df_matrix = pd.DataFrame(vectors.todense(),
                                     columns=vectoriser.get_feature_names_out())
            nlpi.memory_output.append({'data':df_matrix,
                                       'vectoriser':vectoriser})
            
        else:
            
            lst_df = []
            for column in self.subset:
                lvectoriser = clone(vectoriser)
                vectors = vectoriser.fit_transform(list(data[column]))        
                df_matrix = pd.DataFrame(vectors.todense(),
                                         columns=vectoriser.get_feature_names_out())

                nlpi.memory_output.append({'data':df_matrix,
                                       'vectoriser':lvectoriser})
                
            # remove rows
            data.drop(self.subset,axis=1,inplace=True)
            
            # add vectorised data back into data
            
            if(len(lst_df) > 1):
                grouped_labels = pd.concat(lst_df,axis=1)
                add_label = pd.concat([data,grouped_labels],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                          'vectoriser':lvectoriser})
                
            else:
            
                add_label = pd.concat([data,lst_df[0]],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                          'vectoriser':lvectoriser})
            
    
    ''' TF-IDF '''
    
    def tfidf(self,data:pd.DataFrame,args):
            
        pre = {'ngram_range':(1,1),'min_df':1,'max_df':1.0,
               'smooth_idf':True,'use_idf':True}
        
        # create new object
        data = deepcopy(data)
        
        vectoriser = TfidfVectorizer(ngram_range=self.sfp(args,pre,'ngram_range'),
                                     min_df=self.sfp(args,pre,'min_df'),
                                     max_df=self.sfp(args,pre,'max_df'),
                                     tokenizer=args['tokeniser'],
                                     use_idf=self.sfp(args,pre,'use_idf'),
                                     smooth_idf=self.sfp(args,pre,'smooth_idf'))                      
        
        ''' Subset Treatment '''
        
        if(self.subset is None):
        
            data = data.iloc[:,0] # we know it has to be one column 
            vectors = vectoriser.fit_transform(list(data))        
            df_matrix = pd.DataFrame(vectors.todense(),
                                     columns=vectoriser.get_feature_names_out())
            nlpi.memory_output.append({'data':df_matrix,
                                       'vectoriser':vectoriser})
            
        else:
            
            lst_df = []
            for column in self.subset:
                lvectoriser = clone(vectoriser)
                vectors = vectoriser.fit_transform(list(data[column]))        
                df_matrix = pd.DataFrame(vectors.todense(),
                                         columns=lvectoriser.get_feature_names_out())

                lst_df.append(df_matrix)
                
            # remove rows
            data.drop(self.subset,axis=1,inplace=True)
            
            # add vectorised data back into data
            
            if(len(lst_df) > 1):
                grouped_labels = pd.concat(lst_df,axis=1)
                add_label = pd.concat([data,grouped_labels],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                          'vectoriser':lvectoriser})
                
            else:
            
                add_label = pd.concat([data,lst_df[0]],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                       'vectoriser':vectoriser})
        
    ''' Encode a corpus of documents to a numeric tensor '''
                
    def text_torch_encoding(self,data:list,args):
        
        ''' Tokenise Documents '''
        
        lst_tokens = []
        for doc in data:
            lst_tokens.append(word_tokenize(doc))
            
        
        ''' Create dictionary '''
        
        lst_sets = []
        for tokens in lst_tokens:
            lst_sets.append(set(tokens))
        
        corpus_unique_token = set.union(*lst_sets)
        
        # Create a mapping dictionary for all unique tokens in corpus (multiple documents)
        word2id = {token:idx for idx,token in enumerate(corpus_unique_token)}
        vocab_size = len(word2id)
        
        ''' Convert document tokens to numeric value '''
        
        vals = [torch.tensor([word2id[token] for token in document]) for document in lst_tokens]
        padded_vals = pad_sequence(vals).transpose(1,0)
    
        # pad tensor if required    
        if(args['maxlen'] is not None):
            padded_vals = padded_vals[:,:eval(args['maxlen'])]
    
        nlpi.memory_output.append({'data':padded_vals,'dict':word2id})
        
    
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
                                 
                  'tfidf_vectoriser': ['tfidf vectorise',
                                       'tfidf vectorisation',
                                       'tfidf',
                                       'vectorisation using tfidf',
                                       'TfidfVectorizer'],
                                      
                  'torch_text_encode': ['encode documents into tensor',
                                        'encode document into tensor'
                                        'encode corpus into tensor',
                                        'torch encode documents',
                                        'encode documents for torch',
                                        'encode document for torch'
                                        'encode corpus for torch',
                                        'encode text corpus for torch',
                                        'encode document corpus for torch',
                                        'encode text corpus for torch']}

# Other useful information about the task
info_nlpencode = {'encoding_ohe':{'module':'nlp_encoder',
                                  'action':'create encoding',
                                  'topic':'natural language processing',
                                  'subtopic':'create features',
                                  'input_format':'pd.DataFrame',
                                  'output':'data',
                                  'description':'create numerical represention of feature columns containing string names',
                                  'token_compat':'data subset'},
                  
                 'encoding_label':{'module':'nlp_encoder',
                                   'action':'create encoding',
                                   'topic':'natural language processing',
                                   'subtopic':'label encoding',
                                   'input_format':'pd.DataFrame',
                                   'output': 'data vectoriser',
                                   'description':'create numerical presentation of target label containing string names',
                                   'token_compat':'data subset',},
                 
                 'count_vectoriser': {'module':'nlp_encoder',
                                      'action':'create encoding',
                                      'topic':'natural language processing',
                                      'subtopic':'feature generation',
                                      'input_format':'pd.DataFrame',
                                      'output': 'data vectoriser',
                                      'description':'Convert a collection of text documents to a matrix of token counts (unigrams)',
                                      'token_compat':'data subset',
                                      'arg_compat':'ngram_range min_df max_df tokeniser'},
                  
                 'tfidf_vectoriser': {'module':'nlp_encoder',
                                      'action':'create encoding',
                                      'topic':'natural language processing',
                                      'subtopic':'feature generation',
                                      'input_format':'pd.DataFrame',
                                      'output': 'data vectoriser',
                                      'description':'Convert a collection of raw documents to a matrix of TF-IDF features',
                                      'token_compat':'data subset',
                                      'arg_compat':'ngram_range min_df max_df use_idf smooth_idf tokeniser'},

                 'torch_text_encode': {'module':'nlp_encoder',
                                       'action':'create encoding',
                                       'topic':'natural language processing',
                                       'subtopic':'feature generation',
                                       'input_format':'list',
                                       'output':'data',
                                       'description':'Encode documents into numeric representation & add padding to create tensor of identical length',
                                       'token_compat':'data',
                                       'arg_compat': 'maxlen'}
                 }

configure_nlpencoder = {'corpus':dict_nlpencode,'info':info_nlpencode}