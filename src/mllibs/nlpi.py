
from mllibs.nlpm import nlpm
from mllibs.common_corpus import corpus_model
import numpy as np
import pandas as pd
import random
import re
from inspect import isfunction
from seaborn import load_dataset
from mllibs.tokenisers import nltk_wtokeniser,nltk_tokeniser,custpunkttokeniser,n_grams,nltk_wtokeniser_span
from mllibs.data_conversion import convert_to_list,convert_to_df
from string import punctuation


# default plot palette

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

########################################################################

def isfloat(strs:str):
  if(re.match(r'^-?\d+(?:\.\d+)$', strs) is None):
    return False
  else:
    return True


'''

INTERPRETER CLASS (NLPI)

'''
 
class nlpi(nlpm):

    data = {}    # dictionary for storing data
    iter = -1    # keep track of all user requests
    memory_name = []                 # store order of executed tasks
    memory_stack = []                # memory stack of task information
    memory_output = []               # memory output
    model = {}                       # store models
    
    # instantiation requires module
    def __init__(self,module=None,verbose=0):
        self.module = module                  # collection of modules
        self._make_task_info()                # create self.task_info
        self.dsources = {}                    # store all data source keys
        self.token_data = []                  # store all token data
        self.verbose = verbose                # print output text flag
        nlpi.silent = False     
                    
        # class plot parameters
        nlpi.pp = {'alpha':1,'mew':0,'mec':'k','fill':True,'stheme':palette_rgb,'s':30}
        
    # set plotting parameter
        
    def setpp(self,params:dict):
        if(type(params) is not dict):
            if(nlpi.silent is False):
                print("plot parameter dictionary: {'alpha':1,'mew':1,'mec':'k',...}")
        else:
            nlpi.pp.update(params)
            if(nlpi.silent is False):
                print('plot parameter updated!')
   
    @classmethod
    def resetpp(cls):
        nlpi.pp = {'alpha':1,'mew':0,'mec':'k','fill':True,'stheme':palette_rgb,'s':30}

    # Check all available data sources, update dsources dictionary
                    
    def check_dsources(self):
        
        lst_data = list(nlpi.data.keys())            # data has been loaded
        self.dsources = {'inputs':lst_data}
               
        if(nlpi.silent is False): 
            print('inputs:')
            print(lst_data,'\n')
        
        
    ''' 
    
    STORE INPUT DATA
    
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
        data = load_dataset(name)
        self.store_data(data,name,info)

    # [data storage] store active column data (subset of columns)

    def store_data_ac(self,data_name:str,ac_name:str,lst:list):

        if(data_name in nlpi.data):
            if(type(lst) == list):
                nlpi.data[data_name]['ac'][ac_name] = lst
            else:
                print('[note] please use list')

    # [data storage] main data storage function
        
    def store_data(self,data,name:str):
        
		# dictionary to store data information
        datainfo = {'data':None,                      # data storage
                    'subset':None,                    # column subset
                    'splits':None,'splits_col':None,  # row splits (for model) & (for plot)
                    'features':None,'target':None,    # defined features, target variable
                    'cat':None,'num':None,            # names of categorical & numerical columns
                    'miss':None,                      # missing data T/F
                    'size':None,'dim':None,           # dimensions of data
                    'model_prediction':None,          # model prediction values (reg/class)
                    'model_correct':None,             # model prediction T/F (class)
                    'model_error':None,               # model error (reg)
                    'ac': None,                       # active column list (just list of columns)
                    'ft': None                        # feature/target combinations
                    }
    
        ''' 

        Fill out information about dataset 

        '''
                    
        if(isinstance(data,pd.DataFrame)):
            
            ''' [1] Set DataFrame Dtypes '''
            # column names of numerical and non numerical features
                
            datainfo['num'],datainfo['cat'] = self.split_types(data)
            datainfo['ac'] = {}
            datainfo['ac']['num'] = datainfo['num']
            datainfo['ac']['cat'] = datainfo['cat']
            
            ''' [2] Missing Data '''
            # check if there is any missing data

            missing = data.isna().sum().sum()
            
            if(missing > 0):
                datainfo['miss'] = True
            else:
                datainfo['miss'] = False
                
            ''' [3] Column names '''

            datainfo['features'] = list(data.columns)
            
            if(datainfo['target'] is not None):
                datainfo['features'].remove(datainfo['target'])
            
            ''' [4] Determine size of data '''
    
            datainfo['size'] = data.shape[0]
            datainfo['dim'] = data.shape[1]

            # Initialise other storage information
            datainfo['splits'] = {}      # data subset splitting info  (for models)
            datainfo['splits_col'] = {}  #      ""                     (for visualisation - column)
            datainfo['outliers'] = {}    # determined outliers
            datainfo['dimred'] = {}      # dimensionally reduced data 

            datainfo['model_prediction'] = {}
            datainfo['model_correct'] = {}
            datainfo['model_error'] = {}
                
        ''' Store Data '''
        
        if(nlpi.silent is False):
            print(f'\ndata info for {name}')
            print('======================================================')
            print(datainfo)
                 
        datainfo['data'] = data
        nlpi.data[name] = datainfo

        
    '''

    Function to parase model information

    '''
    # required for storing model information

    def store_model(self,model:str,p:str=None,name:str=None):

        if(name is None):
            name = 'model'

        # imported models
        available_models = list(corpus_model.keys())

        # ///////////////////////////////////////////////////////////

        # [A] if p is specified

        # ///////////////////////////////////////////////////////////

        params = {}
        if(p is not None):

            # interpret model parameters p
            # given in " " format

            if('\n' in p):

                splits = p.split('\n')
                splits_clean = list(filter(lambda a: a != "", splits))

                params = {}               # <--- target dictionary
                for opt in splits_clean:

                    key = opt.split('=')[0].strip()
                    value = opt.split('=')[1].strip()

                    try:
                        if('.' in value):
                            value = float(value)
                        else:
                            value = int(value)
                    except:
                        pass
                    
                    params[key] = value

            # given as a list with ,

            else:

                splits = p.split(',')
                
                params = {}
                for parameter in splits:
                    
                    key = parameter.split('=')[0]
                    value = parameter.split('=')[1]

                    try:
                        if('.' in value):
                            value = float(value)
                        else:
                            value = int(value)
                    except:
                        pass
                        
                    params[key] = value

            '''
            
            Interpret Model Input
            
            '''

            # not yet defined parameters, just model name w/o ()

            if("()" not in model or ("(" not in model or ")" not in model)):

                # If model has been written in correct format

                if(model in available_models):
                    if(len(params) != 0):
                        output = model + f"(**{params})"
                    else:
                        output = model + "()"

                # else use model to predict model

                else:
                    print('model not available, predicting model')
                    lmodel = self.module.model['store_model']
                    vectoriser = self.module.vectoriser['store_model']
                    X = vectoriser.transform([model]).toarray()

                    if(len(params) != 0):
                        output = lmodel.predict(X)[0] + f"(**{params})"
                    else:
                        output = lmodel.predict(X)[0] + "()"

                
            # if we have specified parametes p and model mentioned in form ()

            elif("()" in model and p is not None):

                if(model in available_models):
                    if(len(params) != 0):
                        output = model.split('(')[0] + f"(**{params})"
                    else:
                        output = model.predict(X) + "()"

                else:

                    print('model not available, predicting model')
                    lmodel = self.module.model['store_model']
                    vectoriser = self.module.vectoriser['store_model']
                    X = vectoriser.transform([model]).toarray()

                    if(len(params) != 0):
                        output = lmodel.predict(X)[0] + f"(**{params})"
                    else:
                        output = lmodel.predict(X)[0] + "()"

        # [B] if no parameter is given; all data in model string

        elif(p is None):

            # if model is written with (), which may contain parameters

            if('(' in model):

                # remove the brackets
                model_name = model.split('(')[0]

                # model name must be available 

                if(model_name in available_models):

                    # find text inside ()
                    parameters = re.findall(r'\((.*?)\)',model)
                    lst_parameters = parameters[0].split(',')
                    
                    # try to find parameters inside brackets

                    params = {}
                    for parameter in lst_parameters:
                        
                        key = parameter.split('=')[0]
                        value = parameter.split('=')[1]

                        try:
                            if('.' in value):
                                value = float(value)
                            else:
                                value = int(value)
                        except:
                            pass
                            
                        params[key] = value

                    if(len(params) == 0):
                        output = model_name + "()"
                    else:
                        output = model_name + f"(**{params})"

                else:

                    print('model not available, using prediction model')
                    lmodel = self.module.model['store_model']
                    vectoriser = self.module.vectoriser['store_model']
                    X = vectoriser.transform([model_name]).toarray()

                    if(len(params) != 0):
                        output = lmodel.predict(X)[0] + f"(**{params})"
                    else:
                        output = lmodel.predict(X)[0] + "()"

            # if model is wrtten in plain form without ()

            else:

                # check if the base model name is in available models
                # no parameters are available

                # just add brackets
                if(model in available_models):
                    output = model + "()"
                else:
                    print('model not available, using prediction model')
                    lmodel = self.module.model['store_model']
                    vectoriser = self.module.vectoriser['store_model']
                    X = vectoriser.transform([model]).toarray()
                    model_found = lmodel.predict(X)
                    output = model_found[0] + "()"

        nlpi.model[name] = eval(output)
        print('model stored!')

        
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
                'args': self.module_args,
                'ner':self.token_split,
                'seg':self._seg_pred}
     
    '''
    
    NER TAGGING OF INPUT REQUEST
       
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
    
    Check if token names are in data sources 
    
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
            print("[note] input request tokens not found in nlpi.data")

        # check if tokens belong to dataframe column
        self.token_info['column'] = np.nan
        # self.token_info['key'] = np.nan
        # self.token_info['index'] = np.nan

        '''

        Set Token DataFrame Column Association self.token_info['column']

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
    
    Execute user input, have [self.command]
    
    '''
    
    def __getitem__(self,command:str):
        self.exec(command,args=None)
        
    def exec(self,command:str,args:dict=None):                        
        self.do(command=command,args=args)

    '''

    Predict [task_name] using global task classifier

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

        '''

        MAIN PREDICTION

        '''

        predict_module_task(text)
            

    '''

    FILL OUT MODULE ARGUMENTS

    '''

    # if we only have data token, it should be the required function input
        
    def set_moddata(self,in_format,available_data,len_data):

        # number of matches that fit task input format 

        if(len_data == 1):
        
            ldtype = available_data.loc[available_data.index,'dtype'].values[0] # get the data type
            ldata = self.get_td(available_data.index)  # get the data 
            ltoken = list(available_data['token'])
            
            # one token data source which meets function input criteria

            if(ldtype == in_format):
                print('[note] one data source token has been set!')
                self.module_args['data'] = self.get_td(available_data.index)
                self.module_args['data_name'] = ltoken
                
            else:
                
                # try to convert input data to dataframe
                if(in_format == 'pd.DataFrame'):
                    self.module_args['data'] = convert_to_df(ldata)
                elif(in_format == 'list'):
                    self.module_args['data'] = convert_to_list(ldata)
                    
        
        # defining which token to set as data source(s)
            
        elif(len_data > 1):
            
            # match to input requirement
            data_type_match = available_data[available_data['dtype'] == in_format]
            
            # in most cases, there should be only 1 data source passed to funct
            if(len(data_type_match) == 1):
                self.module_args['data'] = self.get_td(data_type_match.index)
                
            # pandas operations can require two (eg. concat)
            elif(len(data_type_match) == 2):
                
                self.module_args['data'] = []
                for idx in list(data_type_match.index):
                    self.module_args['data'].append(self.get_td(idx))
                    
            else:
                
                if(nlpi.silent is False):
                    print('[note] more than 2 data sources found')

                
        else:
            if(nlpi.silent is False):
                print('[note] no data has been set')

    ''' 
    
    [COLUMN SEARCH] DATAFRAME COLUMN SEARCH
    
    '''

    # based on self.token_info [column] data

    def set_modcolumns(self):
            
        # tokenised spans
        tokens_index = nltk_wtokeniser_span(self.command)
        
        # we actually have column (from dataframe) data
        col_data = self.token_info['column'].dropna() # column names
        col_data_idx = col_data.index
        len_col_data = len(col_data)  # number of available column data tokens
        column_tokens = list(self.token_info.loc[col_data_idx,'token'])
        
        if(len_col_data != 0):
            
            # for each token that was found in dataframe columns   
                     
            for token in column_tokens:
                
                # find index where it is located in input command
                idx_match_token = self.command.index(token)
                
                # all possible options we are interested
                lst_options = ['x','y','hue','col','row','target','val']
                
                for option in lst_options:
                    
                    # loop through token index ranges [(0,5),(6-10)...]

                    for ii,segment in enumerate(tokens_index):
                    
                        if(idx_match_token in segment):
                            if(self.tokens[ii-1] == option):
                                self.module_args[option] = token

    '''

    [COLUMN SEARCH] Check Data Storage Content    
    
    '''

    # if set_modcolumns [dataframe columns] does not contain
    # self.data contains additional data about data [such as kfold data]

    def set_gentokens_fromdata(self):

        # cycle through all input request tokens
        for token in self.tokens:

            lst_gtokens = ['x','y','hue','col','row','target','val']

            for gtoken in lst_gtokens:

                # if the previous token was 'hue' etc, check that it exists in data!

                if(self.tokens[self.tokens.index(token)-1] == gtoken):

                    # dictionary of data sources

                    split_col_data = self.data[self.module_args['data_name'][0]]['splits_col']
                    model_pred_data = self.data[self.module_args['data_name'][0]]['model_prediction']
    
                    all_available = []

                    # [a] check columns in data splitting data
                    for key,val in split_col_data.items():
                        all_available.append(key)
                        
                    # [b] check columns in model prediction data 
                    for key,val in model_pred_data.items():
                        for col in list(val.columns):
                            all_available.append(key + "_" + col)
                    
                    # current token 
                    if(token in all_available):
                        self.module_args[gtoken] = token


    '''
    
    USE NER TO SORT MODULE ARGS
    
    '''

    def set_modNER(self,in_format,available_data):
        
        # only if input is a dataframe
        
        if(in_format == 'pd.DataFrame'):       
            
            request_split = self.token_split
            token_split_id = self.token_split_id     
        
            unique_nerid = token_split_id.copy()
            key_token = []
            for lst_tokens in unique_nerid:
                key_token.append([i for i in lst_tokens if i != 4])

            # main function
            def sort_coltoken(tokens:int,lst:list):
                
                # select which key to store column names 
                
                if(0 in tokens):
                    token_name = 'features'
                elif(1 in tokens):
                    token_name = 'target'
                elif(2 in tokens):
                    token_name = 'subset'
                elif(3 in tokens):
                    token_name = None   # data (pass)
                elif(5 in tokens):
                    token_name = 'all'
                else:
                    token_name = None
                               
                # extract token column names
                
                tokens = lst
                bigram_tokens = n_grams(lst,2)
                trigram_tokens = n_grams(lst,3)          
                all_tokens = [tokens,bigram_tokens,trigram_tokens]
                
                # if ner token has been identified 
                
                if(token_name is not None):
                    
                    # classify the action to performed
                    command_document = ' '.join(lst) 
                    pred_name = self.module.test_name('token_subset',command_document)

                    # store tokens which are columns (go through tri,bi,unigrams)
                    
                    column_tokens = []
                    for token_group in all_tokens:
                        for token in token_group:
                            
                            # data origin
                            col_id = self.token_info.loc[token,'column'] 
                            
                            # repeated column token is found in user request 
                            if(type(col_id) is pd.Series or type(col_id) is str):
                                if(token in list(self.module_args['data'].columns)):
                                    column_tokens.append(token)        

                    
                    self._seg_pred.append([token_name,pred_name,column_tokens])                    
                    
                    # if we store only specified/listed tokens 
    
                    if(pred_name == 'only'):
                        
                        self.module_args[token_name] = column_tokens
                        
                    # select all columns in dataframe
                    
                    elif(pred_name == 'all'):
                        
                        all_columns = list(self.module_args['data'].columns)
                        self.module_args[token_name] = all_columns
                        
                    # select all columns but selected column
     
                    elif(pred_name == 'allbut'):
    
                        all_columns = list(self.module_args['data'].columns)
                        remove_columns = column_tokens
                        keep_columns = list(set(all_columns) - set(remove_columns))
                        self.module_args[token_name] = keep_columns
                        
                    # if we need to select numeric columns
                        
                    elif(pred_name == 'numeric'):
                        
                        num,_ = self.split_types(self.module_args['data'])
                        self.module_args[token_name] = num
                        
                    # if we need to select categorical columns
                        
                    elif(pred_name == 'categorical'):
                        
                        _,cat = self.split_types(self.module_args['data'])
                        self.module_args[token_name] = cat

    
                    # subset was stored and added to list
    
                    elif(pred_name == 'fromdata'):
    
                        # match to input requirement
                        lst_match = available_data[available_data['dtype'] == 'list']
    
                        # in most cases, there should be only 1 
                        if(len(lst_match) == 1):
                            self.module_args[token_name] = self.get_td(lst_match.index)
    
                        # pandas operations can require two (eg. concat)
                        elif(len(lst_match) == 2):
    
                            self.module_args[token_name] = []
                            for idx in list(lst_match.index):
                                self.module_args[token_name].append(self.get_td(idx))
    
                            if(nlpi.silent is False):
                                print('stored multiple data in subset, please confirm')
    
                        else:
                            if(nlpi.silent is False):
                                print('please use lists for subset when main data is df')
    
                    else:
                        print('implement me')
                        
                else:
                    
                    # for debug purposes
                    self._seg_pred.append([None,None,None])
    
            # Cycle through all segments split by NER tokens
            
            self._seg_pred = []
            for segment,tokens in zip(key_token,request_split):
                sort_coltoken(segment,tokens)     


    '''

    GENERAL MODULE_ARGS PARAMETER SETTER 

    '''

    def set_gentokens(self):

        for token in self.tokens:
            
            lst_gtokens = ['agg','join','axis','bw','splits','shuffle','rs','const',
                          'threshold','scale','eps','min_samples','ngram_range',
                          'min_df','max_df','n_splits',
                          'use_idf','smooth_idf','dim','window','epoch','lr',
                          'maxlen','sample','whiten','whiten_solver',
                          'n_neighbours','radius','l1_ratio',
                          'alpha_1','alpha_2','lambda_1','lambda_2',
                          'estimator','n_estimators','loss','criterion',
                          'min_samples_leaf','min_samples_split',
                          'max_depth','max_features','bootstrap','oob_score',
                          'max_bins','validation_fraction','n_iter_no_change',
                          'splitter','nan_mode','bootstrap_type','l2_leaf_reg',
                          'col_wrap','kind']
            
            for gtoken in lst_gtokens:
                if(self.tokens[self.tokens.index(token)-1] == gtoken):
                    print('parameter set')
                    self.module_args[gtoken] = token


    ''' MAIN MODULE ARGUMENT '''
    
    def sort_module_args(self):
                
        # input format for the predicted task
        in_format = self.module.mod_summary.loc[self.task_name,'input_format']
            
        # dataframe containing information of data sources of tokens
        available_data = self.token_info[['data','dtype','token']].dropna() 

        # number of rows of data
        len_data = len(available_data)
        
        # operations

        self.set_moddata(in_format,available_data,len_data) # set [module_args['data'],['data_name']]
        # self.set_modcolumns()
        self.set_modNER(in_format,available_data)
        self.set_gentokens()
        # self.set_gentokens_fromdata()                    


    '''
    
    Show module task sumamry   
    
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

    Tokenise Input Command 

    '''

    # set self.tokens
    # set self.token_info dataframe

    def tokenise_request(self):

        # tokenise input, unigram
        self.tokens = custpunkttokeniser(self.command)
        uni = pd.Series(self.tokens).to_frame()
        uni.columns = ['token']
        uni['index_id'] = uni.index
        self.token_info = uni
        self.token_info['type'] = 'uni'
        # self.token_info.index = self.token_info['token']
        # del self.token_info['token']


    '''

    NER for tokens

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

    Single Command Related Operations

    '''

    # find NER tag B-SOURCE

    '''

    NER for [source] (ie. using,for...)

    '''

    def set_NER_source(self,TAG:str):        

        # shifted dataframe data of tagged data
        p2_data = self.token_info[self.token_info['ner_tags'].shift(2) == TAG]
        p1_data = self.token_info[self.token_info['ner_tags'].shift(1) == TAG]
        p0_data = self.token_info[self.token_info['ner_tags'].shift(0) == TAG]

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

        lst_equate = [':',"="]

        # enumerate over all pp tag matches

        for ii,param in enumerate(p0_idx):

            # Try scenario when we have [p+2] token

            try:

                #             TAG    [O]   [O]
                # if we have [main] [p+1] [p+2]
                if(ner_tag_p2[ii] == 'O' and ner_tag_p1[ii] == 'O'):

                    # and [p+1] token is equate token
                    if(value_p1[ii] in lst_equate):
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii],num_idx_id_p2[ii]]
                        self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]
                    else:
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                        self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]
                        print("[note] Two 'O' tags found in a row, choosing nearest value")

                elif(ner_tag_p1[ii] == 'O' and ner_tag_p2[ii] != 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]

                else:
                    print('[note] tag found but parameters not set!')

            except:

                # If [p+2] token doesn't exist

                if(ner_tag_p1[ii] == 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]
                else:
                    print('[note] pp tag found but t+1 tag != O tag')

        
    '''

    NER for [pp] parameters

    '''

    def set_NER_pp(self,TAG:str):        

        # shifted dataframe data of tagged data
        p2_data = self.token_info[self.token_info['ner_tags'].shift(2) == TAG]
        p1_data = self.token_info[self.token_info['ner_tags'].shift(1) == TAG]
        p0_data = self.token_info[self.token_info['ner_tags'].shift(0) == TAG]

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

        lst_equate = [':',"="]

        # enumerate over all pp tag matches

        for ii,param in enumerate(p0_idx):

            # Try scenario when we have [p+2] token

            try:

                #             TAG    [O]   [O]
                # if we have [main] [p+1] [p+2]
                if(ner_tag_p2[ii] == 'O' and ner_tag_p1[ii] == 'O'):

                    # and [p+1] token is equate token
                    if(value_p1[ii] in lst_equate):
                        nlpi.pp[param] = value_p2[ii]
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii],num_idx_id_p2[ii]]
                        self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]
                    else:
                        lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                        nlpi.pp[param] = value_p1[ii]
                        self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]
                        print("[note] Two 'O' tags found in a row, choosing nearest value")

                elif(ner_tag_p1[ii] == 'O' and ner_tag_p2[ii] != 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    nlpi.pp[param] = value_p1[ii]
                    self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]

                else:
                    print('[note] pp tag found but parameters not set!')

            except:

                # If [p+2] token doesn't exist

                if(ner_tag_p1[ii] == 'O'):
                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    nlpi.pp[param] = value_p1[ii]
                    self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]
                else:
                    print('[note] pp tag found but t+1 tag != O tag')
            
         
    # NER surround identifier for parameters in module_args

    def set_NER_params(self,TAG:str):        

        # shifted dataframe data
        p2_data = self.token_info[self.token_info['ner_tags'].shift(2) == TAG]
        p1_data = self.token_info[self.token_info['ner_tags'].shift(1) == TAG]
        p0_data = self.token_info[self.token_info['ner_tags'].shift(0) == TAG]
        p0_idx = list(p0_data['token']) # tokens of identified tags

        # token info (belongs to data)
        columns_p2 = list(p2_data['column'].values)      # is tag a dataframe column
        columns_p1 = list(p1_data['column'].values)      # is tag a dataframe column

        value_p2 = list(p2_data['ttype_storage'].values) # extract token value
        value_p1 = list(p1_data['ttype_storage'].values) # extract token value

        num_idx_id_p2 = list(p2_data['index_id'].values) # numeric indicies
        num_idx_id_p1 = list(p1_data['index_id'].values) # numeric indicies
        num_idx_id_p0 = list(p0_data['index_id'].values) # numeric indicies

        # go through all matches for [TAG]
        # float represents np.nan as all are strings except for NaN

        for ii,param in enumerate(p0_idx):

            if(len(p0_idx) != 0):

                param_id = p0_idx[ii] # NER matches token
                try:
                    p_p2 = columns_p2[ii] # data column info
                    p_p1 = columns_p1[ii] # data column info
                except:
                    p_p2 = np.nan
                    p_p1 = columns_p1[ii] # data column info

                # if token [p+2] belongs to data column 

                if(type(p_p2) is not float):

                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii],num_idx_id_p2[ii]]
                    self.module_args[param_id] = value_p2[ii]
                    self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]

                # if token [p+1] belongs to data column 

                if(type(p_p1) is not float):

                    lst_temp = [num_idx_id_p0[ii],num_idx_id_p1[ii]]
                    self.module_args[param_id] = value_p1[ii]
                    self.mtoken_info = self.mtoken_info[~self.mtoken_info['index_id'].isin(lst_temp)]

    '''

    Single Command Loop

    '''


    def do(self,command:str,args:dict):
       
        # user input command
        self.command = command
        
        # Initialise arguments dictionary
        self.module_args = {'pred_task': None, 'data': None,'subset': None,
                            'splits':None,'features': None, 'target' : None,
                            'x': None, 'y': None, 'hue': None,'col':None,'row':None,
                            'col_wrap':None,'kind':'scatter', 'val':None, 'agg':None,
                            'join':'inner','axis':None,'bw':None,
                            'figsize':[None,None],'test_size':None,
                            'n_splits':None,'shuffle':None,'rs':None,
                            'threshold':None,'eps':None,'min_samples':None,'scale':None,
                            'ngram_range':None,'min_df':None,'max_df':None,
                            'tokeniser':None,'use_idf':None,'smooth_idf':None,
                            'dim':None,'window':None,
                            'epoch':None,'lr':None,'maxlen':None,'const':None,'splitter':None,
                            'neg_sample':None,'batch':None,
                            'kernel':None,'sample':None,'whiten':None,'whiten_solver':None,
                            'n_neighbours':None,'radius':None,'l1_ratio':None,
                            'alpha_1':None,'alpha_2':None,'lambda_1':None,'lambda_2':None,
                            'estimator':None,'n_estimators':None,'loss':None,
                            'criterion':None,'min_samples_leaf':None,'min_samples_split':None,
                            'max_depth':None,'max_features':None,'bootstrap':None,'oob_score':None,
                            'max_bins':None,'validation_fraction':None,'n_iter_no_change':None,
                            'nan_mode':None,'bootstrap_type':None,'l2_leaf_reg':None
                           }
        
        # update argument dictionary if it was set
        if(args is not None):
            self.module_args.update(args)
            
        # Create [self.token_info]

        self.tokenise_request() # tokenise input request
        self.token_NER()        # set [ner_tags] in self.token_info
        self.ner_split()        # ner splitting of request
        self.check_data()       # check tokens for data compatibility
        self.set_token_type()   # find most relevant format for token dtype

        # mtoken is used to in set_NER_xx & relevant tokens are deleted

        self.mtoken_info = self.token_info.copy()
        self.set_NER_params('B-PARAM')
        self.set_NER_pp('B-PP')
        self.set_NER_source('B-SOURCE')

        # text = " ".join(list(self.mtoken_info.index))
        text = " ".join(list(self.mtoken_info['token']))

        # text = self.command
        print('[note] NER used to clean input text!')
        print(text)

        '''

        Task Classification Approaches

        '''

        # self.pred_module_module_task(text) # [module_name] [task_name] prediction 
        self.pred_gtask(text)  # directly predict [task_name]
             
        '''
        
        iterative process
        
        '''
        
        # Iterate if a relevant [task_name] was found

        if(self.task_name is not None):

            nlpi.iter += 1

            # store relevant info into [module_args]
            self.sort_module_args() 
            
            # store activation function information in [module_args]
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
            