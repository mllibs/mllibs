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
from mllibs.dict_helper import sfp, sgp, sfpne, column_to_subset

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
           
    # make selection  

    def sel(self,args:dict):
    
        select = args['pred_task']
        self.data = args['data']
        
        # store subset value from column,col,columns
        args['subset'] = column_to_subset(args)
        self.subset = args['subset'] # multiple column case

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
    # 'subset' or all data
            
    def ohe(self,args:dict):
           
        if(self.subset != None):
            df_matrix = pd.get_dummies(args['data'][args['subset']],dtype='int')
            df_all = pd.concat([args['data'],df_matrix],axis=1)
            nlpi.memory_output.append({'data':df_matrix,'ohe_data':df_all})
        else:
            df_matrix = pd.get_dummies(args['data'],dtype='int')
            nlpi.memory_output.append({'data':df_matrix})
    
    # Label Encode DataFrame column 
    # only 'subset'

    def le(self,args:dict):

        # label encoding requires at least one column
        if(self.subset != None):

            encoder = LabelEncoder()
            vector = encoder.fit_transform(args['data'][args['subset']])
            le_pd = pd.DataFrame(vector,columns=[args['subset'][0] + "_le"])
            df_all = pd.concat([args['data'],le_pd],axis=1)
            df_all.drop(args['subset'],axis=1,inplace=True)
            nlpi.memory_output.append({'data':df_all,'encoder':encoder})

        else:
            print('[note] specify a column you want to label encode]')
                
    ''' 
    
    CountVectoriser of dataframe column
    
    '''

    def cv(self,args:dict):
                    
        # preset value dictionary
        pre = {'ngram_range':(1,1),'min_df':1,'max_df':1.0}

        print(args['subset'])

        if(args['subset'] != None):
        
            vectoriser = CountVectorizer(ngram_range=tuple(sfp(args,pre,'ngram_range')),
                                        min_df=sfp(args,pre,'min_df'),
                                        max_df=sfp(args,pre,'max_df'))

            vectors = vectoriser.fit_transform(args['data'][args['subset'][0]])        
            df_matrix = pd.DataFrame(vectors.todense(),
                                        columns=vectoriser.get_feature_names_out())

            nlpi.memory_output.append({'data':df_matrix,'vectoriser':vectoriser})
                
    ''' 
    
    TF-IDF
    
    '''
    
    def tfidf(self,data:pd.DataFrame,args):
            
        pre = {'ngram_range':(1,1),'min_df':1,'max_df':1.0,'smooth_idf':True,'use_idf':True}
        
        # create new object
        data = deepcopy(data)
        
        vectoriser = TfidfVectorizer(ngram_range=tuple(sfp(args,pre,'ngram_range')),
                                     min_df=sfp(args,pre,'min_df'),
                                     max_df=sfp(args,pre,'max_df'),
                                     tokenizer=args['tokeniser'],
                                     use_idf=sfp(args,pre,'use_idf'),
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
    
