from mllibs.nlpm import nlpm
import numpy as np
import pandas as pd
import random
import re
from inspect import isfunction
import plotly.express as px
import seaborn as sns
from mllibs.tokenisers import custpunkttokeniser
from mllibs.data_conversion import convert_to_list,convert_to_df
from mllibs.df_helper import split_types
from mllibs.str_helper import isfloat
from string import punctuation
from itertools import groupby
import itertools
import difflib
import textwrap

from mllibs.ner_activecolumn import ac_extraction
from mllibs.ner_source import data_tokenfilter

'''

Interpreter class (nlpi)

'''
 
class nlpi(nlpm):

    data = {}                        # dictionary for storing data
    iter = -1                        # keep track of all user requests
    memory_name = []                 # store order of executed tasks
    memory_stack = []                # memory stack of task information
    memory_output = []               # memory output
    model = {}                       # store models
    
    # instantiation requires module
    def __init__(self,module=None):
        self.module = module                  # collection of modules
        self._make_task_info()                # create self.task_info
        self.dsources = {}                    # store all data source keys
        self.token_data = []                  # store all token data
        nlpi.silent = True     
        nlpi.activate = True
        nlpi.lmodule = self.module            # class variable variation for module calls
                    
        # class plot parameters
        nlpi.pp = {'title':None,'template':None,'background':None,'figsize':None}

    '''
    ##############################################################################

                              Plotting parameters nlpi.pp

    ##############################################################################
    '''
        
    # set plotting parameters
    def setpp(self,params:dict):
        if(type(params) is not dict):
            print('[note] such a parameter is not used')
        else:
            nlpi.pp.update(params)
            if(nlpi.silent is False):
                print('[note] plot parameter updated!')

    @classmethod
    def resetpp(cls):
        nlpi.pp = {'title':None,'template':None,'background':None,'figsize':None}

    # Check all available data sources, update dsources dictionary
    def check_dsources(self):
        lst_data = list(nlpi.data.keys())            # data has been loaded
        self.dsources = {'inputs':lst_data}
        
    # [data storage] store active column data (subset of columns)
    def store_ac(self,data_name:str,ac_name:str,lst:list):
        
        if(data_name in nlpi.data):
            if(type(lst) == list):
                nlpi.data[data_name]['ac'][ac_name] = lst
            else:
                print('[note] please use list for subset definition')

    '''
    ###########################################################################

                                    Store Data

    ###########################################################################
    '''

    # [store dataframe] 

    def _store_data_df(self,data,name):

        # dictionary to store data information
        di = {'data':None,                      # data storage
              'subset':None,                    # column subset
              'splits':None,'splits_col':None,  # row splits (for model) & (for plot)
              'features':None,'target':None,    # defined features, target variable
              'cat':None,
              'num':None,            
              'miss':None,                      # missing data T/F
              'size':None,'dim':None,           # dimensions of data
              'model_prediction':None,          # model prediction values (reg/class)
              'model_correct':None,             # model prediction T/F (class)
              'model_error':None,               # model error (reg)
              'ac': None,                       # active column list (just list of columns)
              'ft': None                        # feature/target combinations
              }
        
        ''' [1] Set DataFrame Dtypes '''
        # column names of numerical and non numerical features
            
        di['num'],di['cat'] = split_types(data)
        di['ac'] = {}
        
        ''' [2] Missing Data '''
        # check if there is any missing data

        missing = data.isna().sum().sum()
        
        if(missing > 0):
            di['miss'] = True
        else:
            di['miss'] = False
            
        ''' [3] Column names '''

        di['features'] = list(data.columns)
        
        if(di['target'] is not None):
            di['features'].remove(di['target'])
        
        ''' [4] Determine size of data '''
        di['size'] = data.shape[0]
        di['dim'] = data.shape[1]

        # Initialise other storage information
        di['splits'] = {}      # data subset splitting info  (for models)
        di['splits_col'] = {}  #      ""                     (for visualisation - column)
        di['outliers'] = {}    # determined outliers
        di['dimred'] = {}      # dimensionally reduced data 

        di['model_prediction'] = {}
        di['model_correct'] = {}
        di['model_error'] = {}

        di['data'] = data
        nlpi.data[name] = di

    '''
    ###########################################################################

    Main Function for storing data

    ###########################################################################
    '''
        
    def store_data(self,data,name:str=None):
                    
        # input data cannot be dictionary
        if(name is not None and type(data) is not dict):

            # if dataframe
            if(isinstance(data,pd.DataFrame)):
                column_names = list(data.columns)
                if(name not in column_names):
                    self._store_data_df(data,name)
                else:
                    print(f'[note] please set a different name for {name}')

            # if list
                    
            elif(isinstance(data,list)):
                nlpi.data[name] = {'data':data}

        elif(type(data) is dict):

            # input is a dictionary

            for key,value in data.items():

                if(isinstance(value,pd.DataFrame)):
                    column_names = list(value.columns)

                    if(key not in column_names):
                        self._store_data_df(value,key)
                    else:
                        print(f'[note] please set a different name for data {key}')

                elif(isinstance(value,list)):
                    nlpi.data[key] = {'data':value}
                else:
                    print('[note] only dataframe and lists are accepted')

    # Load Sample Plotly Datasets

    def load_sample_data(self):
        self.store_data(px.data.stocks(),'stocks')
        self.store_data(px.data.tips(),'tips')
        self.store_data(px.data.iris(),'iris')
        self.store_data(px.data.carshare(),'carshare')
        self.store_data(px.data.experiment(),'experiment')
        self.store_data(px.data.wind(),'wind')
        self.store_data(sns.load_dataset('flights'),'flights')
        self.store_data(sns.load_dataset('penguins'),'penguins')
        self.store_data(sns.load_dataset('taxis'),'taxis')
        self.store_data(sns.load_dataset('titanic'),'titanic')
        self.store_data(sns.load_dataset('mpg'),'dmpg')
        if(nlpi.silent is False):
            print('[note] sample datasets have been stored')

    '''

    activation function list

    '''
            
    def fl(self,show='all'):
        if(show == 'all'):
            return self.task_info
        else:
            return dict(tuple(self.task_info.groupby('module')))[show]
     
    '''
    ##############################################################################

    NER TAGGING OF INPUT REQUEST
       
    ##############################################################################
    '''

    # in: self.tokens (required)
    # self.token_split
    # self.token_split_id
    
    def ner_split(self):

        model = self.module.model['token_ner']
        vectoriser = self.module.vectoriser['token_ner']
        X2 = vectoriser.transform(self.tokens).toarray()

        # predict and update self.token_info
        predict = model.predict(X2)
        pd_predict = pd.Series(predict,
                               name='ner_tag',
                               index=self.tokens).to_frame()

        ner_tags = pd.DataFrame({'token':self.tokens,'tag':predict})

        idx = list(ner_tags[ner_tags['tag'] != 4].index)
        l = list(ner_tags['tag'])

        token_split = [list(x) for x in np.split(self.tokens, idx) if x.size != 0]
        token_nerid = [list(x) for x in np.split(l, idx) if x.size != 0]
        
        self.token_split = token_split
        self.token_split_id = token_nerid

       
    ''' 
    ##############################################################################

    Check if token names are in data sources 
    
    ##############################################################################
    '''
	
    # get token data [token_info] -> local self.token_info
    def get_td(self,token_idx:str):
        location = self.token_info.loc[token_idx,'data']
        return self.token_data[int(location)]
    
    # get last result

    def glr(self):
        return nlpi.memory_output[nlpi.iter]     

    # find key matches in [nlpi.data] & [token_info]

    def match_tokeninfo(self):
        dict_tokens = {}
        for source_name in list(nlpi.data.keys()):
            if(source_name in self.tokens):     
                if(source_name in dict_tokens):
                    if(nlpi.silent is False):
                        print('another data source found, overwriting')
                    dict_tokens[source_name] = nlpi.data[source_name]['data']
                else:
                    dict_tokens[source_name] = nlpi.data[source_name]['data']

        return dict_tokens

    def check_data(self):
        
        # intialise data column in token info
        self.token_info['data'] = np.nan  # store data type if present
        self.token_info['dtype'] = np.nan  # store data type if present
        # self.token_info['data'] = self.token_info['data'].astype('Int64')
                    
        # find key matches in [nlpi.data] & [token_info]
        data_tokens = self.match_tokeninfo()

        ''' if we have found matching tokens that contain data '''
                    
        if(len(data_tokens) != 0):

            for (token,value) in data_tokens.items():

                token_index = self.token_info[self.token_info['token'] == token].index
                
                # store data (store index of stored data)
                self.token_info.loc[token_index,'data'] = len(self.token_data) 
                self.token_data.append(value)   
                
                # store data type of found token data

                if(type(value) is eval('pd.DataFrame')):
                    self.token_info.loc[token_index,'dtype'] = 'pd.DataFrame'
                elif(type(value) is eval('pd.Series')):
                    self.token_info.loc[token_index,'dtype'] = 'pd.Series'
                elif(type(value) is eval('dict')):
                    self.token_info.loc[token_index,'dtype'] = 'dict'
                elif(type(value) is eval('list')):
                    self.token_info.loc[token_index,'dtype'] = 'list'   
                elif(type(value) is eval('str')):
                    self.token_info.loc[token_index,'dtype'] = 'str'   
                    
                # # if token correponds to a function; [below not checked!]
                # elif(isfunction(value)):
                #     self.token_info.loc[token_index,'dtype'] = 'function'
                    
                #     for ii,token in enumerate(self.tokens):
                #         if(self.tokens[self.tokens.index(token)-1] == 'tokeniser'):
                #             self.module_args['tokeniser'] = value

        else:
            if(nlpi.silent is False):
                print("[note] input request tokens not found in nlpi.data")

        # check if tokens belong to dataframe column
        self.token_info['column'] = np.nan

        '''
        #######################################################################

        Set Token DataFrame Column Association self.token_info['column']

        #######################################################################
        '''

        # check if tokens match dataframe column,index & dictionary keys
        temp = self.token_info

        # possible multiple dataframe
        dtype_df = temp[temp['dtype'] == 'pd.DataFrame']

        # loop through all rows which are of type DataFrame
        for idx,row in dtype_df.iterrows():

            # get dataframe column names & index

            df_columns = list(self.get_td(idx).columns)
            df_index = list(self.get_td(idx).index)

            # loop through all token variants & see if there are any matches

            tokens_idx = list(temp.index)

            for tidx in tokens_idx:
                token = temp.loc[tidx,'token']
                if(token in df_columns):
                    temp.loc[tidx,'column'] = row.token 
                if(token in df_index):
                    temp.loc[tidx,'column'] = row.token

        # Dictionary

        # dtype_dict = temp[temp['dtype'] == 'dict']

        # for idx,row in dtype_dict.iterrows():

        #     # dictionary keys
        #     dict_keys = list(self.get_td(idx).keys()) # 
        #     tokens = list(temp.index)  # tokens that are dict

        #     for token in tokens:
        #         if(token in dict_keys):
        #             temp.loc[token,'key'] = row.name 
    
        
    ''' 
    ###########################################################################
    
    Execute user input, have [self.command]
    
    ###########################################################################
    '''
    
    def __getitem__(self,command:str):
        self.query(command,args=None)
        
    def query(self,command:str,args:dict=None):                        
        self.do(command=command,args=args)

    def q(self,command:str,args:dict=None):                        
        self.do(command=command,args=args)

    '''
    ###########################################################################

    Predict [task_name] using global task classifier

    ###########################################################################
    '''

    # find the module, having its predicted task 

    def find_module(self,task:str):

        module_id = None
        for m in self.module.modules:
            if(task in list(self.module.modules[m].nlp_config['corpus'].keys())):
                module_id = m

        if(module_id is not None):
            return module_id
        else:
            print('[note] find_module error!')

    # predict global task (sklearn)

    def pred_gtask(self,text:str):
        self.task_name,_ = self.module.predict_gtask('gt',text)
        # having [task_name] find its module
        self.module_name = self.find_module(self.task_name) 

    # predict global task (bert)

    def pred_gtask_bert(self,text:str):
        self.task_name = self.module.predict_gtask_bert('gt',text)
        # having [task_name] find its module
        self.module_name = self.find_module(self.task_name) 

    '''

    # Predict Module Task, set [task_name], [module_name]
    # Two Step Prediction (predict module) then (predict module task)

    '''

    def pred_module_module_task(self,text:str):
        
        # > predict module [module.test_name('ms')]
        # > predict module task 

        # self.module.module_task_name (all tasks in module)

        # Determine which module to activate
        def get_module(text:str):
            ms_name,ms_name_p = self.module.predict_module('ms',text)
            return ms_name,ms_name_p

        # Given [ms_name] (predicted module)
        # Determine which task to activate 

        def get_module_task(ms_name:str,text:str):
            t_pred,t_pred_p = self.module.predict_task(ms_name,text)  
            return t_pred,t_pred_p

        def predict_module_task(text):

            # predict module [ms_name], activation task [t_pred,t_name]
            ms_name,ms_name_p = get_module(text)

            if(ms_name is not None):
                

                # if module passed selection threshold
                t_pred,t_pred_p = get_module_task(ms_name,text)

                if(t_pred is not None):

                    # store predictions
                    self.task_name = t_pred
                    self.module_name = ms_name

                else:
                    self.task_name = None
                    self.module_name = None

            else:
                self.task_name = None
                self.module_name = None

        # MAIN PREDICTION
        predict_module_task(text)
            
    '''
    ##############################################################################

                        Define module_args [data,data_name]

    ##############################################################################  
    '''
    
    def sort_module_args_data(self):
                
        # input format for the predicted task
        in_format = self.module.mod_summary.loc[self.task_name,'input_format']
            
        # dataframe containing information of data sources of tokens
        available_data = self.token_info[['data','dtype','token']].dropna() 

        # number of rows of data
        len_data = len(available_data)

        # check input format requirement
        try:
            in_formats = in_format.split(',')
            in_formats.sort()
        except:
            in_formats = in_format
 
        a_data = list(available_data['dtype'])
        a_data.sort()

        # check compatibility

        if(a_data != in_formats and len(a_data) != 0):
            print('[note] incompatibility in formats!')
            print('in_formats',in_formats)
            print('parsed_data',a_data)

        # input format contains one data source as required by activation function

        if(len_data == 1 and len(in_formats) == 1 and a_data == in_formats):
        
            ldtype = available_data.loc[available_data.index,'dtype'].values[0] # get the data type
            ldata = self.get_td(available_data.index)  # get the data 
            ltoken = list(available_data['token'])
            
            if(nlpi.silent is False):
                print('[note] one data source token has been set!')
            self.module_args['data'] = self.get_td(available_data.index)
            self.module_args['data_name'] = ltoken
                
        elif(len_data == 2 and len(in_formats) == 2 and a_data == in_formats):

            self.module_args['data'] = []; self.module_args['data_name'] = []
            for idx in list(available_data.index):
                self.module_args['data'].append(self.get_td(idx))
                self.module_args['data_name'].append(available_data.loc[idx,'token'])    
                
        else:
            if(nlpi.silent is False):
                print('[note] no data has been set')

    '''
    ###########################################################################

                            Show module task sumamry   
    
    ###########################################################################
    '''
        
    def _make_task_info(self):
        td = self.module.task_dict
        ts = self.module.mod_summary
    
        outs = {}
        for _,v in td.items():
            for l,w in v.items():
                r = random.choice(w)
                outs[l] = r
    
        show = pd.Series(outs,index=outs.keys()).to_frame()
        show.columns = ['sample']
    
        show_all = pd.concat([show,ts],axis=1)

        showlimit = show_all[['module','sample','topic','subtopic','action','input_format',
                              'output','token_compat','arg_compat','description']]
        self.task_info = showlimit
        

    ''' 
    ###########################################################################

                           [ Tokenise Input Command ]

    - set [self.tokens]
    - set [self.token_info] dataframe
    - exclude punctuation from tokens

    ###########################################################################
    '''

    def tokenise_request(self):

        '''
        
        Filter Stop Words
        
        '''

        # don't remove active column punctuation {}
        # {} will be used as active functions registers
        lst = list(punctuation)
        lst.remove('{')
        lst.remove('}')
        lst.remove('-')

        # tokenise input, unigram
        ltokens = custpunkttokeniser(self.command)

        # filter words
        filter_words = ['and','all','a','as']
        tokens = [x for x in ltokens if x not in filter_words]
        
        # remove punctuation
        def remove_punctuation(x):
            return x not in lst

        self.tokens = list(filter(remove_punctuation,tokens))
        self.rtokens = tokens

        '''
        
        Create [self.token_info]

            'token','index_id' & type 'uni' 
            type no longer needed, but implies univariate token
        
        '''

        uni = pd.Series(self.tokens).to_frame()
        uni.columns = ['token']
        uni = uni[~uni['token'].isin(list(lst))].reset_index(drop=True)
        uni['index_id'] = uni.index
        self.token_info = uni
        self.token_info['type'] = 'uni'
        # self.token_info.index = self.token_info['token']
        # del self.token_info['token']

    '''

    Keeper Tokens in main request

        Find which tokens should be kept and not removed
        find all NER tokens (eg. [PARAM]/[SOURCE]) and check 
        if it overlaps with the largest dictionary vocab segment 
        (ie. words which are contained in the training vectoriser dictionary)

        create [keep_token] information in mtoken_info

    '''

    def find_keeptokens(self):

        my_list = list(self.token_info['vocab'])
          
        result = [[i for i, _ in group] for key, group in groupby(enumerate(my_list), key=lambda x: x[1]) if key is True]
        longest_subset = set(max(result,key=len))

        # ner tags which are not O (eg. PARAM/SOURCE)
        notO = [ i for i,j in enumerate(list(self.token_info['ner_tags'])) if j != 'O' ]
        notO_set = set(notO)

        # find overlap between [PARAM] & [SOURCE]
        overlap_idx = longest_subset & notO_set

        self.token_info['keep_token'] = False
        self.token_info.loc[list(overlap_idx),'keep_token'] = True


    '''

    Create NER tags in [self.token_info]

    '''

    # ner inference 
    def token_NER(self):
        self.module.inference_ner_tagger(self.tokens)
        self.token_info['ner_tags'] = self.module.ner_identifier['y_pred']

    # set NER for tokens

    # def token_NER(self):
    #     model = self.module.ner_identifier['model'] 
    #     encoder = self.module.ner_identifier['encoder']
    #     y_pred = model.predict(encoder.transform(self.tokens))
    #     self.token_info['ner_tags'] = y_pred

    # set token dtype [ttype] in [ttype_storage]

    def set_token_type(self):

        lst_types = []; lst_storage = []
        for token in self.tokens:

            if(isfloat(token)):
                type_id = 'float'
                val_id = float(token)
            elif(token.isnumeric()):
                type_id = 'int'
                val_id = int(token)
            else:
                type_id = 'str'
                val_id = str(token)

            lst_types.append(type_id)
            lst_storage.append(val_id)

        self.token_info['ttype'] = lst_types
        self.token_info['ttype_storage'] = lst_storage

    '''
    ##############################################################################

    Check Input Request tokens for function argument compatibility 

    ##############################################################################

    '''

    def set_token_arg_compatibility(self):

        data = list(self.task_info['arg_compat'])
        data_filtered = [i for i in data if i != 'None']
        nested = [i.split(' ') for i in data_filtered]
        unique_args = set([element for sublist in nested for element in sublist])

        # update token_info [argument token]
        self.token_info['token_arg'] = self.token_info['token'].isin(unique_args)

        # update token_info [argument token value]

        ls = self.token_info.copy()
        req_len = len(ls.index)

        param_id = list(ls[ls['token_arg'] == True].index)

        # Column Test

        tcol = ls['column']
        ls['column'] = ls['column'].fillna(0)
        ls['token_argv'] = 0
        for i in param_id:
            for i,row in ls[i+1:req_len].iterrows():
                if(row['column'] != 0):
                    ls.loc[i,'token_argv'] = True
                else:
                    break

        ls['column'] = tcol

        # General 

        for i in param_id:
            for i,row in ls[i+1:req_len].iterrows():
                if(row['ttype'] is not 'str'):
                    ls.loc[i,'token_argv'] = True
                else:
                    break

        for i in param_id:
            ls.loc[i+1,'token_argv'] = True

        # not correct way due to multicolumn input support
        # self.token_info['token_argv'] = self.token_info['token_arg'].shift(1)

        # Add Global Task Vocabulary token information
        lst = list(self.module.vectoriser['gt'].vocabulary_.keys())
        ls['vocab'] = ls['token'].isin(lst)
        self.token_info = ls

    '''
    ###########################################################################

    SUBSET SELECTION BASED ON ACTIVE COLUMNS

    ###########################################################################
    '''

    # subset selection (active columns)
    # can only have one subset per request as we use the last found token ]

    # [note]
    # subsets NEED TO BE USED with ACTIVE COLUMNS
    # but ACTIVE columns can also be used in PARAMS

    @staticmethod
    def set_NER_subset(tdf:pd.DataFrame):

        ls = tdf.copy()
        TAG = ['B-SUBSET','I-SUBSET']
        module_args = {}

        if(ls['ner_tags'].isin(TAG).any()):

            # ac_data dictionary
            ac_data = {}
            for data_name in nlpi.data.keys():
                if(isinstance(nlpi.data[data_name]['data'],pd.DataFrame)):
                    ac_data[data_name] = list(nlpi.data[data_name]['ac'].keys())

            p0_data = ls[ls['ner_tags'].shift(0).isin(TAG)]
            p1_data = ls[ls['ner_tags'].shift(1).isin(TAG)]
            p2_data = ls[ls['ner_tags'].shift(2).isin(TAG)]

            # [note] this won't work for multiple subset matches
            all_window = pd.concat([p0_data,p1_data,p2_data])
            all_window = all_window.drop_duplicates()
            all_idx = list(all_window['index_id'])

            # get only last match 
            p0_data_last = p0_data.iloc[[-1]]
            p1_data_last = p1_data.iloc[[-1]]
            p2_data_last = p2_data.iloc[[-1]]
            v0 = p0_data_last.index_id.values[0]

            # tokens after found subset token
            # need to check if they belong to ac groups
            next_tokens = pd.concat([p0_data_last,p1_data_last,p2_data_last])
            
            next_tokens = next_tokens.drop_duplicates()
            next_tokens = next_tokens.reset_index()
            rhs_idx_window = list(next_tokens['index_id'])

            # tokens to check
            next_token_names = list(next_tokens.loc[1:,'token'].values) 

            # search past tokens for [data token]
            pneg_data_lat = ls.iloc[:v0]
            past_data = pneg_data_lat[pneg_data_lat['dtype'] == 'pd.DataFrame']
            past_data_name = past_data['token'].values[0]
            past_data_columns = ac_data[past_data_name]

            found_overlap = set(next_token_names) & (set(past_data_columns))

            if(len(found_overlap) != 0):
                if(nlpi.silent is False):
                    print(f'[note] specified active function found in LHS data ({past_data_name})')
                store_module_args = nlpi.data[past_data_name]['ac'][found_overlap.pop()]
                module_args['subset'] = store_module_args
                tdf = tdf[~tdf['index_id'].isin(all_idx)]
            else:
                if(nlpi.silent is False):
                    print(f'[note] specified active function NOT found in LHS data ({past_data_name})')        

        return module_args,tdf
        
    '''
    ###########################################################################

    PLOT PARAMETER NER 

        [tdf : self.mtoken_info but is modified in the process]
        [ls : self.mtoken_info @ entry into function]

        - Filtration of self.mtoken_info ( return tdf (modified self.mtoken_info) )
        - nlpi.pp[param] are set in the process 

    ###########################################################################
    '''
    # set nlpi.pp parameters using NER tags and shift window

    @staticmethod
    def filterset_PP(tdf:pd.DataFrame):       

        # input but will always be the same as the input
        ls = tdf

        # shifted dataframe data of tagged data
        p2_data = ls[ls['ner_tags'].shift(2) == "B-PP"]
        p1_data = ls[ls['ner_tags'].shift(1) == "B-PP"]
        p0_data = ls[ls['ner_tags'].shift(0) == "B-PP"]

        # identified pp tokens
        p0_idx = list(p0_data.index) # tokens of identified tags

        # type identified token (token has been stored in correct format it was intended)
        value_p2 = list(p2_data['ttype_storage'].values) # extract token value
        value_p1 = list(p1_data['ttype_storage'].values) # extract token value

        # ner tags for [p+1] [p+2] (eg. TAG, O)
        ner_tag_p2 = list(p2_data['ner_tags'].values) # extract token value
        ner_tag_p1 = list(p1_data['ner_tags'].values) # extract token value

        num_idx_id_p2 = list(p2_data['index_id'].values) # numeric indicies
        num_idx_id_p1 = list(p1_data['index_id'].values) # numeric indicies
        num_idx_id_p0 = list(p0_data['index_id'].values) # numeric indicies

        # equating symbols
        lst_equate = [':',"="]

        # enumerate over all pp tag matches

        for ii,param_idx in enumerate(p0_idx):

            param = p0_data.loc[param_idx,'token']

            try:

                #             TAG    [O]   [O]
                # if we have [main] [p+1] [p+2]
                if(ner_tag_p2[ii] == 'O' and ner_tag_p1[ii] == 'O'):

                    # and [p+1] token is equate token
                    if(value_p1[ii] in lst_equate):
                        nlpi.pp[param] = value_p2[ii]
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii],num_idx_id_p2[ii]]
                        tdf = ls[~ls['index_id'].isin(lst_temp)]
                    else:
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                        nlpi.pp[param] = value_p1[ii]
                        tdf = ls[~ls['index_id'].isin(lst_temp)]
                        if(nlpi.silent is False):
                            print("[note] Two 'O' tags found in a row, choosing nearest value")

                elif(ner_tag_p1[ii] == 'O' and ner_tag_p2[ii] != 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    nlpi.pp[param] = value_p1[ii]
                    tdf = ls[~ls['index_id'].isin(lst_temp)]

                else:
                    if(nlpi.silent is False):
                        print('[note] pp tag found but parameters not set!')

            except:

                # If [p+2] token doesn't exist

                if(ner_tag_p1[ii] == 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    nlpi.pp[param] = value_p1[ii]
                    tdf = ls[~ls['index_id'].isin(lst_temp)]
                else:
                    if(nlpi.silent is False):
                        print('[note] pp tag found but t+1 tag != O tag')

        return tdf
        
    '''
    ###########################################################################

                                  PARAMETER NER

                 [ls] : self.mtoken_info which gets updated
        [module_args] : dict which stores the parameter values

    ###########################################################################
    '''

    # select ner_tag tokens as well as tokens that belong to 
    # goal is to allocate to ner_tag tokens [token] 
    # more compact NER PARAM extractor, can handle multiple columns
    # ignores :/= 

    # need to add double condition for non column PARAM
    # [1] NER tagged as B-PARAM    [2] approved 

    @staticmethod
    def filterset_PARAMS(tdf:pd.DataFrame):

        ls = tdf.copy()
        ls = ls.reset_index(drop=True)
        ls['index_id'] = ls.index
        module_args = {}

        # select rows that belong to data column
        # select = ls[(ls['token_arg'] == True) | ls['ner_tags'].isin(['B-PARAM'])]
        select = ls[~ls['column'].isna() | ls['ner_tags'].isin(['B-PARAM'])]
        select_id = select['index_id']

        # parameter allocation index !(check)

        # selection condition for selecting VALUE in (PARAM - VALUE) pair

        # - a token belonging to a dataframe column
        # - the token is an int or a float
        # - previous token is a defined token_arg

        select_columns = list(ls[ ~ls['column'].isna() | (ls['ttype'].isin(['int','float']) | (ls['token_arg'].shift(1) == True))].index) 

        # parameter source index
        # select_ner_tag = list(ls[~ls['ner_tags'].isin(['O','B-SOURCE'])].index) 
        select_ner_tag = list(ls[ls['ner_tags'].isin(['B-PARAM','I-PARAM'])].index) 

        # set removal constraint: can't remove B-PARAM if it is preceeded by I-SOURCE, B-SOURCE

        # [note]
        # parameter allocation must contain at least one entry
        # parameter allocation can contain more entries than source

        if(len(select_columns) > 0):

            # find the closest minimum value and store it
            closest_minimum_values = []
            for value in select_columns:
                closest_minimum = min(select_ner_tag, key=lambda x: abs(x - value))
                closest_minimum_values.append(closest_minimum)

            remove_idx = []
            remove_idx.extend(select_columns)
            remove_idx.extend(select_ner_tag)
            remove_idx.sort()

            sources = list(ls.loc[closest_minimum_values,'ttype_storage'])
            sources_idx = list(ls.loc[closest_minimum_values,'index_id'])
            sources_map = {'sources':sources,'idx':sources_idx}
            # sources = list(ls.loc[closest_minimum_values,'token'])

            # if [token] is used, need to use str to value conversion for int/float
            # allocation = list(ls.loc[select_columns,'token'])
            allocation = list(ls.loc[select_columns,'ttype_storage'])
            allocation_idx = list(ls.loc[select_columns,'index_id'])
            mapper = dict(zip(allocation,allocation_idx))

            # [my_dict] store PARAM - value combinations
            # [remove tag] decide whether to remove or keep PARAM-value 

            my_dict = {}; remove_tag = {}             
            for value in set(sources):
                my_dict[value] = []      
                remove_tag[value] = None 

            # add values to each parameter to dictionary
            for ii,source in enumerate(sources):
                my_dict[source].append(allocation[ii])

            # store PARAM value
            for key,value in my_dict.items():
                if(len(value) > 1):
                    module_args[key] = value
                elif(len(value) == 1): 
                    module_args[key] = value[0]

            # set removal constraint

            PARAM_IDX = list(ls[ls['ner_tags'].isin(['B-PARAM','I-PARAM'])].index)
            PARAM_TOKEN = list(ls[ls['ner_tags'].isin(['B-PARAM','I-PARAM'])]['token'])

            ls['nts1'] = ls['ner_tags'].shift(1)
            ls['nts2'] = ls['ner_tags'].shift(2)

            # for all matching PARAM cases

            for idx,token in zip(PARAM_IDX,PARAM_TOKEN):

                # remove only conditions
                
                # cond1 = ls['nts1'].isin(['B-SOURCE','I-SOURCE'])
                try:
                    cond2 = ls.loc[idx-1,'token_argv'] == True  # token is token_arg value
                except:
                    cond2 = True 
                    if(nlpi.silent is False):
                        print('[note] parameter has been placed at start, bypassing one condition')

                cond3 = ls['data'].isnull().iloc[idx-1]  # data is NULL (no data)
                cond4 = ls['column'].isnull().iloc[idx-1]  # column is NULL (no data)

                if(cond2 and cond3 or not cond4):
                    remove_tag[token] = 'remove'
                else:
                    remove_tag[token] = 'keep'

            # create group based indicies

            my_dict_mapped = dict.fromkeys(my_dict.keys())
            for key,value in my_dict.items():
                my_dict_mapped[key] = list(map(mapper.get, value))

            # loop through keep/remove dictionary

            # for key,value in remove_tag.items():

            #     if(value is 'remove'):
            #         my_dict_mapped[key].append(min(my_dict_mapped[key]) - 1)
            #         ls = ls[~ls['index_id'].isin(my_dict_mapped[key])]

            # remove indicies from [remove_idx] if condition [keep_token] is met
            keep_idx = list(ls[ls['keep_token'] == True].index)

            if(len(keep_idx)>0):
                remove_idx = [value for index, value in enumerate(remove_idx) if value not in keep_idx]

            # remove tokens associated with PARAMS
            ls = ls[~ls['index_id'].isin(remove_idx)]

        else:
            if(nlpi.silent is False):
                print('[note] no parameters to extract (possible NER miss)')

        # return stored [self.module_args, self.mtoken_info]
        return module_args, ls
    
    '''
    ###########################################################################

                                  Logical Filters

    ###########################################################################
    '''

    # Filter base request before classification
    # request can't end with a preposition

    def preposition_filter(self):

        prepositions = [
            'about','above','across','after','against','along','among','around',
            'as','at','before','behind','below','beneath','beside','between',
            'beyond','by','down','during','for','from','in','inside','into',
            'near','of','off','on','onto','out','outside','over','past','through',
            'throughout','to','towards','under','underneath','until','up','with','within'
        ]

        tls = self.mtoken_info

        last = None
        found = True
        while found == True:
            for i,j in tls[::-1].iterrows():
                if(j['token'] not in prepositions):
                    found = False
                    last = i + 1
                    break

        if(last != None):
            self.mtoken_info = tls[0:last]

    # function which after having predicted an [activation function] 
    # checks if input data requirement : has the data been set?
        
    def check_data_compatibility(self):

        def type_to_str(inputs):
            if(isinstance(inputs,eval('pd.DataFrame')) == True):
                return 'pd.DataFrame'
            elif(isinstance(inputs,eval('pd.Series')) == True):
                return 'pd.Series'
            elif(isinstance(inputs,eval('list')) == True):
                return 'list'
            elif(inputs is None):
                return 'None'

        # input format as string format
        input_data = type_to_str(self.module_args['data'])

        # check input function data requirement
        # task = self.module_args['pred_task'] # the set task (not yet available)
        task = self.task_name
        input_format_str = self.task_info.loc[task,'input_format'] 

        if(input_data != input_format_str):
            nlpi.activate = False
            print('[note] data input does not coincide with af requirement!')
        
    
    '''
    ##############################################################################

    Initialise module_args dictionary

    ##############################################################################
    '''

    def initialise_module_args(self):

        # Initialise arguments dictionary (critical entries)
        self.module_args = {'pred_task': None, 
                            'data': None,'data_name':None,
                            'subset': None,
                            'features': None, 'target' : None}

        # (update) Activation Function Parameter Entries 
        data = list(self.task_info['arg_compat'])
        data_filtered = [i for i in data if i != 'None']
        nested = [i.split(' ') for i in data_filtered]
        unique_args = set([element for sublist in nested for element in sublist])

        for val in unique_args:
            self.module_args[val] = None
          
    '''
    #######################################################################
  
  
                             [ do Single Iteration ]

                               used with query, q 
    
    
    #######################################################################
    '''

    def do(self,command:str,args:dict):
       
        # user input command
        self.command = command
        
        # initialise self.module_args
        self.initialise_module_args()

        # update argument dictionary (if it was set manually)
        if(args is not None):
            self.module_args.update(args)
            
        '''
        #######################################################################

                               create self.token_info
    
        #######################################################################
        '''
            
        # tokenise input query 
        self.tokenise_request() # tokenise input request

                                    # create [self.token_info]

        # define ner tags for each token
        self.token_NER()        # set [ner_tags] in self.token_info

                                # set:

                                    # self.token_info['ner_tags']

        self.check_data()       # check tokens for data compatibility

                                # set:

                                    # self.token_info['data']
                                    # self.token_info['dtype']
                                    # self.token_info['column']
                                    
        self.set_token_type()   # find most relevant format for token dtype
        
                                # set:
                                
                                    # self.token_info['ttype']
                                    # self.token_info['ttype_storage'] 
                                    
                                    # converted token type (eg. str -> int)
                                        
        self.set_token_arg_compatibility()  # determine function argument compatibility
        
                                    # self.token_info['arg_compat']
                                    
        self.find_keeptokens()
        
                                    # self.token_info['keep_token']
      
        '''
        #######################################################################
  
                            [  Updated NER approach ]
    
            Updated approach utilises inserted tokens to describe the token

                - [self.module_args] has been initialised
                - [self.token_info] has been created

                    - new NER doesn't really use it:
                      only for column, data, token, ner_tag
  
        #######################################################################
        '''      
        
        # df_tinfo - will be used to remove rows that have been filtered
        df_tinfo = self.token_info.copy()

        print('\n##################################################################\n')
        print('[note] extracting parameters from input request!\n')
      
        print(f"[note] input request:")
        print(textwrap.fill(' '.join(list(df_tinfo['token'])), 60))
        print('')
      
        # extract and store active column (from older ner)
        tmod_args,df_tinfo = ac_extraction(df_tinfo,nlpi.data)      
        self.module_args.update(tmod_args)
      
        # find the difference between two strings 
        # using split() return the indicies of tokens which are missing
        
        def string_diff_index(ref_string:str,string:str):
          
          # Tokenize both strings
          reference_tokens = ref_string.split()
          second_tokens = string.split()
          
          # Find the indices of removed tokens
          removed_indices = []
          matcher = difflib.SequenceMatcher(None, reference_tokens, second_tokens)
          for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'delete':
              removed_indices.extend(range(i1, i2))
              
          return removed_indices
      
        '''
        
        1. Add -data token information into request
        
        '''
      
        def create_labels_data(tokens:list,data_id:list):
          
          # Add a new token "-data" before tokens with non-"O" data_id
          tokenized_string = []
          for token, id in zip(tokens, data_id):
            if id != -1:
              tokenized_string.extend(['-data', token])
            else:
              tokenized_string.append(token)
              
          # Join the tokenized string back together
          result = ' '.join(tokenized_string)
          
          # Remove "and" between two "-data" words
          result = re.sub(r'(-data \w+) and (-data \w+)', r'\1 \2', result)
          
          return result
      
        '''
        
        2. Using -data tokens store the relevant data & simple filter
        
        '''
        
        # general [-] tag storage
        def filter_set_token(user_request:str):
          
          tokens = user_request.split()
          parameter_prefix = "-"
          base_request_tokens = []
          parameters = {}
          i = 0
          while i < len(tokens):
            token = tokens[i]
            
            if token.startswith(parameter_prefix):
              parameter = token[1:]
              if i + 1 < len(tokens) and not tokens[i + 1].startswith(parameter_prefix):
                value = tokens[i + 1]
                if parameter in parameters:
                  # If the parameter already exists in the dictionary, convert its value to a list
                  if isinstance(parameters[parameter], list):
                    parameters[parameter].append(value)
                  else:
                    parameters[parameter] = [parameters[parameter], value]
                else:
                  parameters[parameter] = value
                i += 1
              else:
                parameters[parameter] = None
            else:
              base_request_tokens.append(token)
            i += 1
            
          base_request = " ".join(base_request_tokens)
          
          return base_request, parameters
        
        # required information
        token_id = list(df_tinfo['token'])
        data_id = list(df_tinfo['data'].fillna(-1).astype('int'))
        
        # tag data based tokens and store & filter
        labelled_str = create_labels_data(token_id,data_id)        
        filtered_request,data_parameters = filter_set_token(labelled_str)
        
        # determine which idx was removed
        removed_idx = string_diff_index(" ".join(token_id),filtered_request)
        
#       print('1. input data tokens')
#       print(" ".join(token_id))
      
#       print('1. labelled string')
#       print(labelled_str)
#       print('1. output')
#       print("filtered request:", filtered_request)
#       print("data parameters:", data_parameters)
#       print('')
#       print('')
        
        # update 
        df_tinfo = df_tinfo.drop(removed_idx)
        df_tinfo = df_tinfo.reset_index(drop=True)
        
        '''

        3. Add -values / -column token information into request

        '''

        def create_labels_param(tokens:list,column_id:list,ner_id:list):
            
            # Add "-column" token before tokens based on column_id
            new_tokens = []
            for token, col_id, ner in zip(tokens, column_id, ner_id):
                if col_id != 'O':
                    new_tokens.append("-column")
                if ner.lower() in ['b-param', 'i-param']:
                    new_tokens.append("~" + token)
                else:
                    new_tokens.append(token)
                
            # Join the tokens back into a string
            output_string = ' '.join(new_tokens)
            
            # Tokenize the string
            tokens = output_string.split()
            
            # Add "-value" token before the index containing a float or an integer
            new_tokens = []
            for token in tokens:
                if re.match(r'^[-+]?[0-9]*\.?[0-9]+$', token):
                    new_tokens.append("-value")
                new_tokens.append(token)
                
            # Join the tokens back into a string
            output_string = ' '.join(new_tokens)
            
            return output_string
        
        '''
        
        4. Storing ner parameters that are columns/values from dataframe
        
        '''
        
        # not used at the moment (replaced by dictionary updater 
        def determine_number_type(string):
          if string.isdigit():
            return int(string)
          try:
            return float(string)
          except ValueError:
            return string
          
        def ner_column_parsing(request:str):
          
          # Remove "and" between two "-column" words
          request = re.sub(r'(-column \w+) and (-column \w+)', r'\1 \2', request)
          
          # Tokenize the request by splitting on whitespace
          tokens = request.split()
          
          # Initialize an empty dictionary
          param_dict = {}
          
          # Initialize an empty list to store filtered tokens
          filtered_tokens = []
          filter_idx = []
          # Loop through the tokens
          for i in range(len(tokens)):
            token = tokens[i]
            
            # Check if the token starts with "-column"
            if token.startswith("-column"):
              
              # Find the nearest token containing "~" to the left
              for j in range(i-1, -1, -1):
                if "~" in tokens[j]:
                  filter_idx.append([i for i in range(j,i+2)])
                  # Store the next token after "-column" in a list
                  column_value = param_dict.get(tokens[j], [])
                  column_value.append(tokens[i+1])
                  param_dict[tokens[j]] = column_value
                  break
                
                # Check if the token starts with "-value"
                
            elif(token.startswith("-value")):
              
              # Find the nearest token containing "~" to the left
              for j in range(i-1, -1, -1):
                if "~" in tokens[j]:
                  filter_idx.append([i for i in range(j,i+2)])
                  # Store the next token after "-column" in a list
                  column_value = param_dict.get(tokens[j], [])
                  column_value.append(tokens[i+1])
                  param_dict[tokens[j]] = column_value
                  break
                
            else:
              # Add non-key or non-value tokens to filtered_tokens list
              filtered_tokens.append(token)
              
          if(bool(param_dict)):
            
            # index of tokens to be removed
            grouped_lists = {}
            for sublist in filter_idx:
              first_value = sublist[0]
              last_value = sublist[-1]
              if first_value not in grouped_lists or last_value > grouped_lists[first_value][-1]:
                grouped_lists[first_value] = sublist
                
            selected_lists = list(grouped_lists.values())
            selected_lists = list(itertools.chain.from_iterable(selected_lists))
            filtered_tokens = [token for index, token in enumerate(tokens) if index not in selected_lists]
            
            # Iterate over the dictionary and remove it from brackets if list contains only one entry
            for key, value in param_dict.items():
              # Check if the length of the value list is 1
              if len(value) == 1:
                # Extract the single value from the list and update the dictionary
                param_dict[key] = value.pop()
                
          else:
            print('[note] no ner parameter filtration and extraction was made')
            filtered_tokens = tokens
            
          # Create a new dictionary with keys without the ~
          new_dict = {key[1:]: value for key, value in param_dict.items()}	
            
          return new_dict," ".join(filtered_tokens)

        # required information
        token_id = list(df_tinfo['token'])
        column_id = list(df_tinfo['column'].fillna('O'))
        ner_id = list(df_tinfo['ner_tags'])
      
        labelled_str = create_labels_param(token_id,column_id,ner_id)
        param_dict,filtered_request = ner_column_parsing(labelled_str)
        removed_idx = string_diff_index(" ".join(token_id),filtered_request)
        
        # update param_dict (change string to int/float if needed)
#       param_dict = {key: [float(val) if val.replace('.', '', 1).isdigit() else val for val in value] if isinstance(value, list) else float(value) if value.replace('.', '', 1).isdigit() else value for key, value in param_dict.items()}
        
        # update param_dict (change string to int/float if needed)
        for key, value in param_dict.items():
          if isinstance(value, list):
            param_dict[key] = [float(x) if '.' in x else int(x) if x.isdigit() else x for x in value]
          else:
            if '.' in value:
              param_dict[key] = float(value)
            else:
              param_dict[key] = int(value) if value.isdigit() else value
        
#       print(param_dict)

#       print('2. input column string')
#       print(" ".join(token_id))
#       print('2. labelled string')
#       print(labelled_str)
#       print('2. output')
#       print("filtered request:", filtered_request)
#       print("column parameters:", param_dict)
#       print('')
#       print('')
      
        
        # update 
        df_tinfo = df_tinfo.drop(removed_idx)
        df_tinfo = df_tinfo.reset_index(drop=True)
        
        '''
        
        5. Set column labels again to identify subset columns that didnt have ner param token
        
          for storage, we can use generic function [filter_set_token]
        
        '''
      
        def label_subset(tokens:str,column_id:list):
          
          # Add "-column" and "~" tokens before tokens based on column_id and ner_id
          new_tokens = []
          for token, col_id in zip(tokens, column_id):
            if col_id != 'O':
              new_tokens.append("-column")
            new_tokens.append(token)
            
          # Join the tokens back into a string
          output_string = ' '.join(new_tokens)
          
          return output_string
      
        # required information
        token_id = list(df_tinfo['token'])
        column_id = list(df_tinfo['column'].fillna('O'))
        
        labelled_str = label_subset(token_id,column_id)
        filtered_request,subset_parameters = filter_set_token(labelled_str)
        
        # determine which idx was removed
        removed_idx = string_diff_index(" ".join(token_id),filtered_request)
      
#       print('3. input subset string')
#       print(" ".join(token_id))
#       print('3. labelled string')
#       print(labelled_str)
#       print('3. output')
#       print("filtered request:", filtered_request)
#       print("column parameters:", subset_parameters)
#       print('')
#       print('')
        
        # update 
        df_tinfo = df_tinfo.drop(removed_idx)
        df_tinfo = df_tinfo.reset_index(drop=True)
        
        '''
        
        6. remove [token_remove] tokens
        
        '''
        
        # required information
        token_id = list(df_tinfo['token'])
        ner_id = list(df_tinfo['ner_tags'].fillna('O'))
        
        # remove [token_remove] tokens
        def remove_tokens(tokens:list,ner_id:list):
          result = [tokens[i] for i in range(len(tokens)) if ner_id[i].lower() not in ['b-token_remove', 'i-token_remove']]
          return " ".join(result)
      
        filtered_request = remove_tokens(token_id,ner_id)
        removed_idx = string_diff_index(" ".join(token_id),filtered_request)
        
#       print('4. output')
#       print("filtered request:", filtered_request)
#       print('')
#       print('')
        
        # update 
        df_tinfo = df_tinfo.drop(removed_idx)
        df_tinfo = df_tinfo.reset_index(drop=True)
        
        '''
        
        Preposition Filter ( + custom word removal )

            if ner doesn't remove the [token_remove] tokens correctly
            prepositions can accumulate at the end of a request

            eg. "create plotly scatter plot using set"
                                              -    -

            remove them from the end until no more is found in [prepositions]
          
        
        '''
    
        def preposition_filter(token_info:pd.DataFrame):
          
            prepositions = [
            'about','above','across','after','against','along','among','around',
            'as','at','before','behind','below','beneath','beside','between',
            'beyond','by','down','during','for','from','in','inside','into',
            'near','of','off','on','onto','out','outside','over','past','through',
            'throughout','to','towards','under','underneath','until','up','with',
            'within','set','using']
          
            tls = token_info.copy()
          
            last = None
            found = True
            while found == True:
                for i,j in tls[::-1].iterrows():
                    if(j['token'] not in prepositions):
                        found = False
                        last = i + 1
                        break
                  
            if(last != None):
                token_info = tls[0:last]
                
            return token_info
                
        # remove prepositions (update df_tinfo directly)
        df_tinfo = preposition_filter(df_tinfo)
  
        filtered = " ".join(list(df_tinfo['token']))
        print('\n[note] filtered request:')
        print(filtered)
      
        '''
        
        Store data into [self.module_args]
        
        '''
      
#       print('module_args')
#       print(self.module_args)
#       print('stored data:')
#       print(data_parameters)
        self.module_args['data'] = nlpi.data[data_parameters['data']]['data']
        self.module_args['data_name'] = data_parameters['data']
        self.module_args.update(param_dict)
        self.module_args.update(subset_parameters)
      
      
        print('\n##################################################################\n')
        
        '''
        #######################################################################
        
              Data Extraction & Filter (Older NER [based on token_info]) 
        
        #######################################################################
        '''      

#       self.mtoken_info = self.token_info.copy()
#
#       # extract and store active column
#       tmod_args,self.mtoken_info = ac_extraction(self.mtoken_info,nlpi.data)      
#       self.module_args.update(tmod_args)
#
#       # extract and store data sources 
#       self.mtoken_info = data_tokenfilter(self.mtoken_info)    
#
#       # filter out PP tokens + store PP param (in nlpi.pp)
#       self.mtoken_info = self.filterset_PP(self.mtoken_info)     
#
        # extract and store PARAM data
#       tmod_args,self.mtoken_info = self.filterset_PARAMS(self.mtoken_info)   
#       self.module_args.update(tmod_args)
#
#       tmod_args, self.mtoken_info = self.set_NER_subset(self.mtoken_info)
#       self.module_args.update(tmod_args)
#
#       self.preposition_filter() # final preposition filter
#
#       before = " ".join(self.rtokens)
#       filtered = " ".join(list(self.mtoken_info['token']))
#
#       if(nlpi.silent is False):
#           print('\n[note] NER used to clean input text!')
#           print('[input]')
#           print(before)
#           print('[after]')
#           print(filtered,'\n')

        '''
        #######################################################################
  
                                Text Classification 
    
            Having filtered and extracted data from input request, classify
  
        #######################################################################
        '''      

        # 1] predict module

        # self.task_name, self.module_name prediction
        # self.pred_module_module_task(text) 
        
        # 2] global activation function task prediction
        
        self.pred_gtask(filtered)      # directly predict [self.task_name]
        # self.pred_gtask_bert(filtered) # directly predict [self.task_name]
                         
        '''
        #######################################################################
        
                            Iterative Step Loop Preparation
        
        #######################################################################
        '''      
            
        if(self.task_name is not None):

            # Store activation function information in module_args [pred_task]
            
            self.module_args['pred_task'] = self.task_name

            # store task name information
            
            self.module_args['task_info'] = self.task_info.loc[self.task_name]

            # store data related
            
                      # - self.module_args['data'],
                      # - self.module_args['data_name']
                      
#           self.sort_module_args_data()  

            # check compatibility between predict activation function data
            # data requirement & the extracted data type
            
            self.check_data_compatibility()
        
        # Iterate if a relevant [task_name] was found

        if(nlpi.activate is True):

            if(self.task_name is not None):

                nlpi.iter += 1
                            
                # store iterative data
                nlpi.memory_name.append(self.task_name)  
                nlpi.memory_stack.append(self.module.mod_summary.loc[nlpi.memory_name[nlpi.iter]] )
                nlpi.memory_info = pd.concat(self.memory_stack,axis=1) # stack task information order
                
                # activate function [module_name] & pass [module_args]
                self.module.modules[self.module_name].sel(self.module_args)
            
                if(len(nlpi.memory_output) == nlpi.iter+1):
                    pass
                else:
                    nlpi.memory_output.append(None) 
                
        else:
            print('[note] no iteration activated!')

        nlpi.activate = True

    '''
    
    Manually Call Activation functions
    
    '''

    def miter(self,module_name:str,module_args:dict):
        nlpi.iter += 1
        self.module.modules[module_name].sel(module_args)

    # reset nlpi session

    def reset_session(self):
        nlpi.iter = -1
        nlpi.memory_name = []
        nlpi.memory_stack = []
        nlpi.memory_output = []
      