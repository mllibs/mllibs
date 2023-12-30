from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from mllibs.nlpi import nlpi
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from sklearn.base import clone
from copy import deepcopy
import torch
from mllibs.tokenisers import nltk_tokeniser
from torch.nn.utils.rnn import pad_sequence
from mllibs.nlpm import parse_json
import pkg_resources
import json

'''

Encoding Text Data

'''
class encoder(nlpi):
    
    def __init__(self):
        self.name = 'encoder'
        path = pkg_resources.resource_filename('mllibs', '/nlp/mencoder.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)

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

    # store subset column names into a single value
    # subsets can originate from [col,column]
    # certain acivation functions use subsets

    def check_subset(self,args:dict):

        self.subset = None
        if args['column'] is not None:
            self.subset = args['column']
        if args['col'] is not None:
            self.subset = args['col']
           
    # make selection  

    def sel(self,args:dict):
    
        select = args['pred_task']
        self.data = args['data']
        self.check_subset(args)
        
        ''' select appropriate predicted method '''
        
        if(select == 'encoding_ohe'):
            self.ohe(args)
        elif(select == 'encoding_label'):
            self.le(args)
        elif(select == 'count_vectoriser'):
            self.cv(args)  
        elif(select == 'tfidf_vectoriser'):
            self.tfidf(args)
        elif(select == 'torch_text_encode'):
            self.text_torch_encoding(args)
            
    # One Hot Encode DataFrame 
    # input dataframe 
            
    def ohe(self,args:dict):
           
        if(self.subset != None):
            df_matrix = pd.get_dummies(args['data'])
        else:
            ldf = args['data'][self.subset]
            df_matrix = pd.get_dummies(ldf)
            df_all = pd.concat([args['data'],df_matrix],axis=1)
    
        nlpi.memory_output.append({'data':df_matrix,'ohe_data':df_all})
                   
        
    # Label Encode DataFrame column 
    # input dataframe

    def le(self,args:dict):
        
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
                
   
    ''' 
    
    CountVectoriser 
    
    '''

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
            
            # add vectorised data b ack into data
            
            if(len(lst_df) > 1):
                grouped_labels = pd.concat(lst_df,axis=1)
                add_label = pd.concat([data,grouped_labels],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                          'vectoriser':lvectoriser})
                
            else:
            
                add_label = pd.concat([data,lst_df[0]],axis=1)
                nlpi.memory_output.append({'data':add_label,
                                          'vectoriser':lvectoriser})
            
    
    ''' 
    
    TF-IDF
    
    '''
    
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
        
    ''' 
    
    Encode a corpus of documents to a numeric tensor 
    
    '''
                
    def text_torch_encoding(self,data:list,args):
        
        ''' Tokenise Documents '''
        
        lst_tokens = []
        for doc in data:
            lst_tokens.append(nltk_tokeniser(doc))
            
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
    
