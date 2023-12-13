
from mllibs.nlpm import nlpm
import numpy as np
import pandas as pd
import random
import re
from inspect import isfunction
import plotly.express as px
import seaborn as sns
from mllibs.tokenisers import nltk_wtokeniser,nltk_tokeniser,custpunkttokeniser,n_grams,nltk_wtokeniser_span
from mllibs.data_conversion import convert_to_list,convert_to_df
from string import punctuation

# default plot palette
def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0,2,4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

########################################################################

def isfloat(strs:str):
  if(re.match(r'^-?\d+(?:\.\d+)$', strs) is None):
    return False
  else:
    return True


'''
##############################################################################

INTERPRETER CLASS (NLPI)

##############################################################################
'''
 
class nlpi(nlpm):

    data = {}    # dictionary for storing data
    iter = -1    # keep track of all user requests
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
        nlpi.lmodule = self.module            # class variable variation for module calls
                    
        # class plot parameters
        nlpi.pp = {'alpha':None,'mew':None,'mec':None,'fill':None,'stheme':palette_rgb,'s':None, 'title':None,'template':None,'background':None}

    # set plotting parameter
        
    def setpp(self,params:dict):
        if(type(params) is not dict):
            print('[note] such a parameter is not used')
        else:
            nlpi.pp.update(params)
            if(nlpi.silent is False):
                print('[note] plot parameter updated!')

    @classmethod
    def resetpp(cls):
        nlpi.pp = {'alpha':1,'mew':0,'mec':'k','fill':True,'s':30,'title':None}

    # Check all available data sources, update dsources dictionary
                    
    def check_dsources(self):
        
        lst_data = list(nlpi.data.keys())            # data has been loaded
        self.dsources = {'inputs':lst_data}
               
        # if(nlpi.silent is False): 
        #     print('inputs:')
        #     print(lst_data,'\n')
        
        
    ''' 
    ##############################################################################
    
    STORE INPUT DATA
    
    ##############################################################################
    '''
    
    # split dataframe columns into numeric and categorical
    
    @staticmethod
    def split_types(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
        numeric = df.select_dtypes(include=numerics)
        categorical = df.select_dtypes(exclude=numerics)
        return list(numeric.columns),list(categorical.columns)

    # Load Dataset from Seaborn Repository    
    def load_dataset(self,name:str,info:str=None):
        # load data from seaborn repository             
        data = sns.load_dataset(name)
        self.store_data(data,name,info)


    '''
    ##############################################################################

    Store Active Columns (in nlp)

    ##############################################################################
    '''

    # [data storage] store active column data (subset of columns)

    def store_ac(self,data_name:str,ac_name:str,lst:list):

        if(data_name in nlpi.data):
            if(type(lst) == list):
                nlpi.data[data_name]['ac'][ac_name] = lst
            else:
                print('[note] please use list for subset definition')

    '''
    ##############################################################################

    Store Data

    ##############################################################################
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
            
        di['num'],di['cat'] = self.split_types(data)
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

    # Main Function for storing data
        
    def store_data(self,data,name:str=None):
                    
        if(name is not None and type(data) is not dict):

            if(isinstance(data,pd.DataFrame)):
                self._store_data_df(data,name)
            elif(isinstance(data,list)):
                nlpi.data[name] = {'data':data}

        elif(type(data) is dict):

            for key,value in data.items():

                if(isinstance(value,pd.DataFrame)):
                    self._store_data_df(value,key)
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
        self.store_data(sns.load_dataset('mpg'),'mpg')
        if(nlpi.silent is False):
            print('[note] sample datasets have been stored')

    # activation function list
    def fl(self,show='all'):
                            
        # function information
        df_funct = self.task_info
        
        if(show == 'all'):
            return df_funct
        else:
            return dict(tuple(df_funct.groupby('module')))[show]
        
    # debug, show information
    def debug(self):
        
        return {'module':self.module.mod_summary,
                'token': self.token_info,
                'args': self.module_args}
     
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

        ner_tags = pd.DataFrame({'token':self.tokens,
                                 'tag':predict})

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
	
    # get token data
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


    # check if self.tokens is in active column
    
    # def check_ac(self):

    
    def check_data(self):
        
        # intialise data column in token info
        self.token_info['data'] = np.nan  # store data type if present
        self.token_info['dtype'] = np.nan  # store data type if present
        # self.token_info['data'] = self.token_info['data'].astype('Int64')
                    
        # find key matches in [nlpi.data] & [token_info]
        data_tokens = self.match_tokeninfo()
                    
        ''' if we have found matching tokens that contain data'''
                    
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
        # self.token_info['key'] = np.nan
        # self.token_info['index'] = np.nan

        '''
        ##############################################################################

        Set Token DataFrame Column Association self.token_info['column']

        ##############################################################################
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

        # # Dictionary

        # dtype_dict = temp[temp['dtype'] == 'dict']

        # for idx,row in dtype_dict.iterrows():

        #     # dictionary keys
        #     dict_keys = list(self.get_td(idx).keys()) # 
        #     tokens = list(temp.index)  # tokens that are dict

        #     for token in tokens:
        #         if(token in dict_keys):
        #             temp.loc[token,'key'] = row.name 
    
        
    ''' 
    ##############################################################################
    
    Execute user input, have [self.command]
    
    ##############################################################################
    '''
    
    def __getitem__(self,command:str):
        self.exec(command,args=None)
        
    def exec(self,command:str,args:dict=None):                        
        self.do(command=command,args=args)

    '''
    ##############################################################################

    Predict [task_name] using global task classifier

    ##############################################################################
    '''

    def pred_gtask(self,text:str):

        # predict global task

        def get_globaltask(text:str):
            gt_name,gt_name_p = self.module.predict_gtask('gt',text)
            return gt_name,gt_name_p

        self.task_name,_ = get_globaltask(text)

        # having [task_name] find its module

        def find_module(task:str):

            module_id = None
            for m in self.module.modules:
                if(task in list(self.module.modules[m].nlp_config['corpus'].keys())):
                    module_id = m

            if(module_id is not None):
                return module_id
            else:
                print('[note] find_module error!')

        self.module_name = find_module(self.task_name) 
  
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

    Define module_args [data,data_name]

    '''
    
    def sort_module_args_data(self):
                
        # input format for the predicted task
        in_format = self.module.mod_summary.loc[self.task_name,'input_format']
            
        # dataframe containing information of data sources of tokens
        available_data = self.token_info[['data','dtype','token']].dropna() 

        # number of rows of data
        len_data = len(available_data)
        
        # operations
        # set [module_args['data'],['data_name']]

        # check input format requirement
        try:
            in_formats = in_format.split(',')
            in_formats.sort()
        except:
            in_formats = in_format
 
        a_data = list(available_data['dtype'])
        a_data.sort()

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
    ##############################################################################

    Show module task sumamry   
    
    ##############################################################################
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
    ##############################################################################

                             [[ Tokenise Input Command ]]

    ##############################################################################
    '''

    # set self.tokens
    # set self.token_info dataframe
    # exclude punctuation from tokens

    def tokenise_request(self):

        # {} will be used as active functions registers
        lst = list(punctuation)
        lst.remove('{')
        lst.remove('}')
        lst.remove('-')

        # tokenise input, unigram
        tokens = custpunkttokeniser(self.command)
        self.rtokens = tokens

        # remove punctuation
        def remove_punctuation(x):
            return x not in lst

        self.tokens = list(filter(remove_punctuation,tokens))

        uni = pd.Series(self.tokens).to_frame()
        uni.columns = ['token']
        uni = uni[~uni['token'].isin(list(lst))].reset_index(drop=True)
        uni['index_id'] = uni.index
        self.token_info = uni
        self.token_info['type'] = 'uni'
        # self.token_info.index = self.token_info['token']
        # del self.token_info['token']

    '''
    ##############################################################################

    NER for tokens

    ##############################################################################
    '''

    # set NER tokenisation

    def token_NER(self):
        model = self.module.ner_identifier['model'] 
        encoder = self.module.ner_identifier['encoder']
        y_pred = model.predict(encoder.transform(self.tokens))
        self.token_info['ner_tags'] = y_pred

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

    Check Input Request tokens for function argument compatibility 

    ##############################################################################

    '''

    def set_token_arg_compatibility(self):

        lst_data = []
        compat_sets = {}
        data = list(self.task_info['arg_compat'])
        data_filtered = [i for i in data if i != 'None']
        nested = [i.split(' ') for i in data_filtered]
        unique_args = set([element for sublist in nested for element in sublist])

        # update token_info
        self.token_info['token_arg'] = self.token_info['token'].isin(unique_args)

    '''

    ACTIVE COLUMN NER PARSING

    ##############################################################################
    '''

    # active columns need to be found and removed first
    # also subset and active columns are not the same
    # subset - selection of a subset of columns 
    # active columns - group of column names, can be used for any operation 

    def ac_extraction(self):

        ls = self.token_info
        lst = list(ls['token'])

        # helper function to get bracket pairs
        def get_bracket_content(lst):
            stack = []
            pairs = []
            for i, item in enumerate(lst):
                if item == '{':
                    stack.append(i)
                elif item == '}':
                    if stack:
                        pairs.append((stack.pop(), i))
                    else:
                        print("Error: Unmatched closing bracket at index", i)
            if stack:
                print("Error: Unmatched opening bracket(s) at index", stack)
            
            return pairs

        # find all bracket pairs in list
        pairs = get_bracket_content(lst)

        lst_act_functions = []
        for pair in pairs:
            select_col_content = ls.iloc[pair[0]:pair[1]+1]
            act_funct = select_col_content[~select_col_content['token'].isin(['{','}'])]
            lst_act_functions.append(list(act_funct['token'])[0])

        ### FIND WHICH AC TO STORE 

        # find what the active columns were assigned to by selecting the first 
        # for each pair select the closest param [ner_tag]
        lst_param_idx = []
        for pair in pairs:
            ldf = ls[(ls.index < pair[0]) & (ls['ner_tags'].isin(['B-PARAM']))].reset_index(drop=True)
            previous_token_idx = ldf.iloc[-1]['index_id']
            lst_param_idx.append(previous_token_idx)

        # save the [param_id] identifier
        lst_param_id = []
        for i in lst_param_idx:
            lst_param_id.append(ls.iloc[i]['token'])

        ### REMOVE THEM 

        # select everything in between [ner_tag] and last BRACKET
        remove_idx = []
        for ii,pair in enumerate(pairs):
            remove_idx.append(list(ls.iloc[lst_param_idx[ii]:pair[1]+1]['index_id']))

        for pair in remove_idx:
            ls = ls[~(ls.index_id.isin(pair))]

        # update mtoken_info
        self.mtoken_info = ls
        self.mtoken_info = self.mtoken_info.reset_index(drop=True)
        self.mtoken_info['index_id'] = list(self.mtoken_info.index)

        ### FIND ALL ACTIVE COLUMN KEYS
        # this is needed to store AC 

        # ac_data dictionary

        ac_data = {}
        for data_name in nlpi.data.keys():
            if(isinstance(nlpi.data[data_name]['data'],pd.DataFrame)):
                ac_data[data_name] = list(nlpi.data[data_name]['ac'].keys())

        # return the data name 

        def find_ac(name):
            data_name = None
            for key,values in ac_data.items():
                if(name in values):
                    data_name = key

            if(data_name is not None):
                return data_name
            else:
                return None

        ### INSERT ACTIVE COLUMN INTO RELEVANT PARAM

        # d_id - {d_id}
        # ac_id - param_id token identifier

        for d_id,ac_id in zip(lst_act_functions,lst_param_id):
            data_name = find_ac(d_id)

            if(data_name != None):
                print(f'[note] storing [{ac_id}] in module_args')
                self.module_args[ac_id] = nlpi.data[data_name]['ac'][d_id]

    '''

    SUBSET SELECTION BASED ON ACTIVE COLUMNS

    ##############################################################################
    '''

    # subset selection (active columns)
    # can only have one subset per request as we use the last found token ]

    # [note]
    # subsets NEED TO BE USED with ACTIVE COLUMNS
    # but ACTIVE columns can also be used in PARAMS

    def set_NER_subset(self,TAG=['B-SUBSET','I-SUBSET']):

        ls = self.mtoken_info.copy()

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
                self.module_args['subset'] = store_module_args
                self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(all_idx)]
            else:
                if(nlpi.silent is False):
                    print(f'[note] specified active function NOT found in LHS data ({past_data_name})')        

    '''

    SOURCE NER EXTRACTION
           
    ##############################################################################
    '''

    # find all the SOURCE related tokens that need to removed and remove them
    # from self.token_info uses NER tags for B-SOURCE/I-SOURCE 

    def data_extraction(self):

        # identify source related index and remove them
        ls = self.mtoken_info

        # number of data sources
        nsources = len(ls[~ls['data'].isna()])

        # either there are multiple or only a single one
        try:
            max_lendiff = int(np.max(ls[ls['dtype'].notna()]['index_id'].diff()))
        except:
            max_lendiff = 0

        # number of data sources > 0 to activate

        if(nsources > 0):

            # ONE SOURCE CASE

            if(max_lendiff == 0):

                if(nlpi.silent is False):
                    print('[note] one source token format')
                lst_remove_idx = []
                
                # get the data index id
                p0 = list(ls[ls['dtype'].notna()]['index_id'])[0]
                lst_remove_idx.append(p0)

                ''' CHECKING PREVIOUS TOKENS '''
                
                # create window to check previous tokens
                pm1 = p0 - 1; pm2 = p0 - 2

                # [test1] check if previous token is in punctuation
                punct_test = list(ls.loc[pm1,'token'])[0] in punctuation
                # [test2] check if previous token belongs to SOURCE token
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = ls.loc[pm2,'ner_tags'] in ['B-SOURCE','I-SOURCE']

                if(punct_test and source_test_pm2):
                    lst_remove_idx.append(pm1) # remove punctuation token
                    lst_remove_idx.append(pm2) # remove SOURCE token
                elif(source_test_pm1):
                    lst_remove_idx.append(pm1) # remove SOURCE token
                else:
                    pass # nothing needs to be removed

            elif(max_lendiff == 1):
                if(nlpi.silent is False):
                    print('[note] two sources tokens side by side format')
                # get data index id
                lst_remove_idx = list(ls[ls['dtype'].notna()]['index_id'])
                p0 = lst_remove_idx[0] # first index only

                ''' CHECKING PREVIOUS TOKENS '''

                # create window to check previous tokens
                pm1 = p0 - 1; pm2 = p0 - 2
                # [test1] check if previous token is in punctuation
                punct_test = list(ls.loc[pm1,'token'])[0] in punctuation
                # [test2] check if previous token belongs to SOURCE token
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = ls.loc[pm2,'ner_tags'] in ['B-SOURCE','I-SOURCE']

                if(punct_test and source_test_pm2):
                    lst_remove_idx.append(pm1) # remove punctuation token
                    lst_remove_idx.append(pm2) # remove SOURCE token
                elif(source_test_pm1):
                    lst_remove_idx.append(pm1) # remove SOURCE token
                else:
                    pass # nothing needs to be removed

            elif(max_lendiff == 2):
                if(nlpi.silent is False):
                    print('[note] two sources separated by a single token format')
                lst_remove_idx = list(ls[ls['dtype'].notna()]['index_id'])    
                lst_remove_idx.append(lst_remove_idx[0] + 1)
                p0 = lst_remove_idx[0] # first index only

                ''' CHECKING PREVIOUS TOKENS '''

                # create window to check previous tokens
                pm1 = p0 - 1; pm2 = p0 - 2
                # [test1] check if previous token is in punctuation
                punct_test = list(ls.loc[pm1,'token'])[0] in punctuation
                # [test2] check if previous token belongs to SOURCE token
                source_test_pm1 = ls.loc[pm1,'ner_tags'] in ['B-SOURCE','I-SOURCE']
                source_test_pm2 = ls.loc[pm2,'ner_tags'] in ['B-SOURCE','I-SOURCE']

                if(punct_test and source_test_pm2):
                    lst_remove_idx.append(pm1) # remove punctuation token
                    lst_remove_idx.append(pm2) # remove SOURCE token
                elif(source_test_pm1):
                    lst_remove_idx.append(pm1) # remove SOURCE token
                else:
                    pass # nothing needs to be removed

            else:
                if(nlpi.silent is False):
                    print('[note] multiple sources w/ distance > 2 found (error)')

            # update token_info
            self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_remove_idx)]
        
    '''

    PLOT PARAMETER NER

    ##############################################################################
    '''
    # set nlpi.pp parameters using NER tags and shift window

    def filterset_PP(self,TAG:str='B-PP'):       

        ls = self.mtoken_info

        # shifted dataframe data of tagged data
        p2_data = ls[ls['ner_tags'].shift(2) == TAG]
        p1_data = ls[ls['ner_tags'].shift(1) == TAG]
        p0_data = ls[ls['ner_tags'].shift(0) == TAG]

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
                        self.mtoken_info = ls[~ls['index_id'].isin(lst_temp)]
                    else:
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                        nlpi.pp[param] = value_p1[ii]
                        self.mtoken_info = ls[~ls['index_id'].isin(lst_temp)]
                        if(nlpi.silent is False):
                            print("[note] Two 'O' tags found in a row, choosing nearest value")

                elif(ner_tag_p1[ii] == 'O' and ner_tag_p2[ii] != 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    nlpi.pp[param] = value_p1[ii]
                    self.mtoken_info = ls[~ls['index_id'].isin(lst_temp)]

                else:
                    if(nlpi.silent is False):
                        print('[note] pp tag found but parameters not set!')

            except:

                # If [p+2] token doesn't exist

                if(ner_tag_p1[ii] == 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    nlpi.pp[param] = value_p1[ii]
                    self.mtoken_info = ls[~ls['index_id'].isin(lst_temp)]
                else:
                    if(nlpi.silent is False):
                        print('[note] pp tag found but t+1 tag != O tag')
            

    '''
    ##############################################################################

    PARAMETER NER SETTERS

    ##############################################################################
    '''

    # select ner_tag tokens as well as tokens that belong to 
    # goal is to allocate to ner_tag tokens [token] 
    # more compact NER PARAM extractor, can handle multiple columns
    # ignores :/= 

    # need to add double condition for non column PARAM
    # [1] NER tagged as B-PARAM    [2] approved 

    def filterset_PARAMS(self):

        ls = self.mtoken_info.copy()

        # select rows that belong to data column
        # select = ls[(ls['token_arg'] == True) | ls['ner_tags'].isin(['B-PARAM'])]
        select = ls[~ls['column'].isna() | ls['ner_tags'].isin(['B-PARAM'])]
        select_id = select['index_id']

        # parameter allocation index !(check)

        # selection condition:
        # - a token belonging to a dataframe column
        # - the token is an int or a float
        # - previous token is a defined token_arg

        select_columns = list(ls[ ~ls['column'].isna() | (ls['ttype'].isin(['int','float']) | (ls['token_arg'].shift(1) == True))].index) 
        # select_columns = list(ls[ ~ls['column'].isna() | (ls['token_arg'].diff(1) is True)].index) 

        # parameter source index
        select_ner_tag = list(ls[~ls['ner_tags'].isin(['O','B-SOURCE'])].index) 

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

            sources = list(ls.loc[closest_minimum_values,'token'])
            allocation = list(ls.loc[select_columns,'token'])

            # set module_args

            my_dict = {}  # Empty dictionary to store lists
            for value in set(sources):
                my_dict[value] = []  # Create an empty list for each value

            for ii,source in enumerate(sources):
                my_dict[source].append(allocation[ii])

            for key,value in my_dict.items():
                if(len(value) > 1):
                    self.module_args[key] = value
                elif(len(value) == 1): 
                    self.module_args[key] = value[0]

            # remove tokens associated with PARAMS
            self.mtoken_info = ls[~ls['index_id'].isin(remove_idx)]

        else:
            if(nlpi.silent is False):
                print('[note] no parameters to extract')

    '''
    ##############################################################################

    Single Command Loop

    ##############################################################################
    '''

    def initialise_module_args(self):

        # Initialise arguments dictionary (critical entries)
        self.module_args = {'pred_task': None, 'data': None,'subset': None,
                            'features': None, 'target' : None}


        # (update) Activation Function Parameter Entries 
        lst_data = []
        compat_sets = {}
        data = list(self.task_info['arg_compat'])
        data_filtered = [i for i in data if i != 'None']
        nested = [i.split(' ') for i in data_filtered]
        unique_args = set([element for sublist in nested for element in sublist])
        # print(unique_args)

        for val in unique_args:
            self.module_args[val] = None

    def do(self,command:str,args:dict):
       
        # user input command
        self.command = command
        
        # initialise self.module_args
        self.initialise_module_args()

        # update argument dictionary (if it was set manually)
        if(args is not None):
            self.module_args.update(args)
            
        # Create [self.token_info]

        self.tokenise_request() # tokenise input request
        self.token_NER()        # set [ner_tags] in self.token_info

                                # set:

                                    # self.token_info['ner_tags']

        # self.ner_split()        # ner splitting of request

                                # create:

                                   # create self.token_split
                                   # create self.token_split_id

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

        self.mtoken_info = self.token_info.copy()

        self.ac_extraction()      # extract and store active column
        self.data_extraction()    # extract and store data sources 
        self.filterset_PP()       # filter out PP tokens + store PP param (in nlpi.pp)
        self.filterset_PARAMS()   # extract and store PARAM data
        self.set_NER_subset()  

        before = " ".join(self.rtokens)
        after = " ".join(list(self.mtoken_info['token']))

        if(nlpi.silent is False):
            print('\n[note] NER used to clean input text!')
            print('[input]')
            print(before)
            print('[after]')
            print(after,'\n')

        '''

        Task Classification Approaches

        '''

        # self.pred_module_module_task(text) # [module_name] [task_name] prediction 
        self.pred_gtask(after)  # directly predict [task_name]
             
        '''

        [[ Iterative process ]]
        
        '''
        
        # Iterate if a relevant [task_name] was found

        if(self.task_name is not None):

            nlpi.iter += 1

            # Store module_args [data,data_name]
            self.sort_module_args_data() 
            
            # Store activation function information in module_args [pred_task]
            self.module_args['pred_task'] = self.task_name
            
            # store iterative data
            nlpi.memory_name.append(self.task_name)  
            nlpi.memory_stack.append(self.module.mod_summary.loc[nlpi.memory_name[nlpi.iter]] )
            nlpi.memory_info = pd.concat(self.memory_stack,axis=1) # stack task information order
            
            # activate function [module_name] & pass [module_args]
            self.module.modules[self.module_name].sel(self.module_args)
            
            # if not data has been added
            # initialise output data (overwritten in module.function
            
            if(len(nlpi.memory_output) == nlpi.iter+1):
                pass
            else:
                nlpi.memory_output.append(None) 


    def reset_session(self):
        nlpi.iter = 0
        nlpi.memory_name = []
        nlpi.memory_stack = []
