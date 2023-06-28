
from mllibs.nlpm import nlpm
import numpy as np
import pandas as pd
import random
import panel as pn
from nltk.tokenize import word_tokenize, WhitespaceTokenizer 
from inspect import isfunction
from seaborn import load_dataset

# default plot palette

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

########################################################################


# interaction & tect interpreter class
 
class nlpi(nlpm):

    data = {}    # dictionary for storing data
    iter = -1    # keep track of all user requests
    memory_name = []                 # store order of executed tasks
    memory_stack = []                # memory stack of task information
    memory_output = []
    
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
    
    store data 
    
    '''
    
    # split dataframe columns into numeric and categorical
    
    @staticmethod
    def split_types(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
        numeric = df.select_dtypes(include=numerics)
        categorical = df.select_dtypes(exclude=numerics)
        return list(numeric.columns),list(categorical.columns)

        
    ''' 
    
    STORE INPUT DATA
    
    '''

    # Load Dataset from Seaborn Repository
    
    def load_dataset(self,name:str,info:str=None):
        
        # load data from seaborn repository             
        data = load_dataset(name)
        self.store(data,name,info)
        
    def store(self,data,name:str,info:str=None):
        
		# dictionary to store data information
        datainfo = {'data':None,'subset':None,'splits':None,
                    'features':None,'target':None,
                    'cat':None,'num':None,
                    'miss':None,'corpus':None,
                    'size':None,'dim':None,
                    }
    
                    
        ''' 1. DESCRIPTION TOKENISATION '''
                    
        if(info is not None):
            
            dtokens = self.nltk_wtokeniser(info) # unigram
            dbgtokens = self.n_grams(dtokens,2) # bigram
            
            
            # store target variable token name
            
            if(isinstance(data,pd.DataFrame)):
                
                ''' Find Target Variable Tokens '''
                
                # no check is done to confirm it exists in data
            
                # loop through bigram tokens
                
                for ii,token in enumerate(dbgtokens):
                    if(dbgtokens[dbgtokens.index(token)-1] == 'target variable'):
                        datainfo['target'] = token

                if(datainfo['target'] is None):
                    
                    # loop through tokens
                
                    for ii,token in enumerate(dtokens):
                        if(dtokens[dtokens.index(token)-1] == 'target'):
                            if(datainfo['target'] is None):
                                datainfo['target'] = token
                            
                    
        ''' 2. Fill out information about dataset '''
                    
        if(isinstance(data,pd.DataFrame)):
            
            ''' Set DataFrame Dtypes '''
            # column names of numerical and non numerical features
                
            datainfo['num'],datainfo['cat'] = self.split_types(data)
            
            ''' Missing Data '''
            # check if we have missing data
            
            missing = data.isna().sum().sum()
            
            if(missing > 0):
                datainfo['miss'] = True
            else:
                datainfo['miss'] = False
                
            ''' Column names '''
                
            datainfo['features'] = list(data.columns)
            
            if(datainfo['target'] is not None):
                datainfo['features'].remove(datainfo['target'])

            ''' Check Multicollinearity '''
                
            

            
            ''' Check for corpus columns '''
                
            # a corpus column is defined as a column with tokenisable text of length > 5
                
            col_corpus = []
            for col in data.columns:
                token_len = [] 
                for ii,row in enumerate(data[col]):
                    if(ii<5):
                        try:
                            token_len.append(len(WhitespaceTokenizer().tokenize(row)))
                        except:
                            pass
                        
                tokens = len([*filter(lambda x: x >= 5, token_len)]) > 0
                if(tokens):
                    col_corpus.append(col)
                        
            datainfo['corpus'] = col_corpus  
            
            ''' Determine size of data '''
    
            datainfo['size'] = data.shape[0]
            datainfo['dim'] = data.shape[1]

            # Initialise other storage information
            datainfo['splits'] = {}   # data subset splitting info
            datainfo['outliers'] = {}  # determined outliers
            datainfo['dimred'] = {}    # dimensionally reduced data 
                
        ''' Store Data '''
        
        if(nlpi.silent is False):
            print(f'\ndata information for {name}')
            print('=========================================')
            print(datainfo)
                 
        datainfo['data'] = data
        nlpi.data[name] = datainfo
        
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
    
    # define self.token_split, self.token_split_id
    
    def ner_split(self):

        model = self.module.model['token_ner']
        vectoriser = self.module.vectoriser['token_ner']
        X2 = vectoriser.transform(self.tokens).toarray()

        # predict and update self.token_info
        predict = model.predict(X2)
        pd_predict = pd.Series(predict,name='ner_tag',index=self.tokens).to_frame()
        labels = self.tokens
        ner_tags = pd.DataFrame({'token':labels,'tag':predict})

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
	
    def get_td(self,token):
        location = self.token_info.loc[token,'data']
        return self.token_data[int(location)]
    
    # get last result
    
    def glr(self):
        return nlpi.memory_output[nlpi.iter]     
    
    def check_data(self):
        
        # intialise data column in token info
        self.token_info['data'] = np.nan  # store data type if present
        self.token_info['dtype'] = np.nan  # store data type if present
        self.token_info['data'] = self.token_info['data'].astype('Int64')
                    
        # cycle through all available key names    
        dict_tokens = {}
        for source_name in list(nlpi.data.keys()):
            if(source_name in self.tokens):     
                if(source_name in dict_tokens):
                    if(nlpi.silent is False):
                        print('another data source found, overwriting')
                    dict_tokens[source_name] = nlpi.data[source_name]['data']
                else:
                    dict_tokens[source_name] = nlpi.data[source_name]['data']
                    
        ''' if we have found matching tokens '''
                    
        if(len(dict_tokens) != 0):
            for token,value in dict_tokens.items():
                
                # store data (store index of stored data)
                self.token_info.loc[token,'data'] = len(self.token_data) 
                self.token_data.append(value)   
                
                # store data type
                if(type(value) is eval('pd.DataFrame')):
                    self.token_info.loc[token,'dtype'] = 'pd.DataFrame'
                elif(type(value) is eval('pd.Series')):
                    self.token_info.loc[token,'dtype'] = 'pd.Series'
                elif(type(value) is eval('dict')):
                    self.token_info.loc[token,'dtype'] = 'dict'
                elif(type(value) is eval('list')):
                    self.token_info.loc[token,'dtype'] = 'list'   
                elif(type(value) is eval('str')):
                    self.token_info.loc[token,'dtype'] = 'str'   
                    
                # if token correponds to a function; 
                elif(isfunction(value)):
                    self.token_info.loc[token,'dtype'] = 'function'
                    
                    for ii,token in enumerate(self.tokens):
                        if(self.tokens[self.tokens.index(token)-1] == 'tokeniser'):
                            self.module_args['tokeniser'] = value
                
                
        else:
            pass
        
        # check if tokens belong to dataframe column
        self.token_info['column'] = np.nan
        self.token_info['key'] = np.nan
        self.token_info['index'] = np.nan

        ''' Check Inside '''
        # check if tokens match dataframe column,index & dictionary keys
        temp = self.token_info
        
        # possible multiple dataframe
        ldfs = temp[temp['dtype'] == 'pd.DataFrame']

        # i - token name; j token dataframe
        for i,j in ldfs.iterrows():

            # df column & index names
            df_columns = list(self.get_td(i).columns)
            df_index = list(self.get_td(i).index)

            tokens = list(temp.index)
            for token in tokens:
                if(token in df_columns):
                    temp.loc[token,'column'] = j.name 
                if(token in df_index):
                    temp.loc[token,'column'] = j.name

        ldfs = temp[temp['dtype'] == 'dict']

        for i,j in ldfs.iterrows():

            # dictionary keys
            dict_keys = list(self.get_td(i).keys())
            tokens = list(temp.index)

            for token in tokens:
                if(token in dict_keys):
                    temp.loc[token,'key'] = j.name 
                        
    @staticmethod
    def n_grams(tokens,n):
        lst_bigrams = [' '.join(i) for i in [tokens[i:i+n] for i in range(len(tokens)-n+1)]]
        return lst_bigrams
    
        
    ''' 
    
    Execute user input 
    
    '''
    
    def __getitem__(self,command:str):
        self.exec(command,args=None)
        
    def exec(self,command:str,args:dict=None):                        
        self.do(command=command,args=args)
  
    '''
    
    Execute everything relevant for single command 
    
    '''
    
    def do_predict(self):
        

        '''
        
        Predict both Module & Activation Task 
        
        '''
            
        def predict_module_task(text):
    
            # determine which module to activate
            ms_name = self.module.test_name('ms',text)
            if(nlpi.silent is False):
                print(f'using module: {ms_name}')
        
            # Available tasks 
            lst_tasks = self.module.module_task_name[ms_name]
            t_pred_p = self.module.test(ms_name,text)  
            t_pred = np.argmax(t_pred_p)
    
            # [2] name o the module task to be called
            t_name = lst_tasks[t_pred]
            if(nlpi.silent is False): 
                print(f'Executing Module Task: {t_name}')

            # store predictions
            self.task_pred = t_pred
            self.task_name = t_name
            self.module_name = ms_name
        
        # check condition 
        if(len(self.token_split)>1):
            text = ' '.join(self.token_split[0])
            if(nlpi.silent is False):
                print('ner split token found, using base text for task prediction')
                print(f"{text}")
        else:
            text = self.command
        
        # predict both module and task
        predict_module_task(text)
            
    
    @staticmethod
    def convert_to_df(ldata):
        
        if(type(ldata) is list or type(ldata) is tuple):
            return pd.Series(ldata).to_frame()
        elif(type(ldata) is pd.Series):
            return ldata.to_frame()
        else:
            raise TypeError('Could not convert input data to dataframe')
            
    @staticmethod
    def convert_to_list(ldata):
        
        if(type(ldata) is str):
            return [ldata]
        else:
            raise TypeError('Could not convert input data to list')
    
    def sort_module_args(self):
                
        # input format 
        in_format = self.module.mod_summary.loc[self.task_name,'input_format']
            
        # dataframe containing information of data sources of tokens
        available_data = self.token_info[['data','dtype']].dropna() 
        len_data = len(available_data) # number of rows of data
        
        '''
        
        DATA TOKEN IS FOUND 
        
        //////////////////////////////////////////////////////////////////
        
        '''
        
        # if we only have data token, it should be the required function input
        
        if(len_data == 1):
        
            ldtype = available_data.loc[available_data.index,'dtype'].values[0] # get the data type
            ldata = self.get_td(available_data.index)  # get the data 
            
            # We have one data source & it meets input function criteria
            # else try to change data 

            if(ldtype == in_format):
                self.module_args['data'] = self.get_td(available_data.index)
                self.module_args['data_name'] = available_data.index
                
            else:
                
                # try to convert input data to dataframe
                if(in_format == 'pd.DataFrame'):
                    self.module_args['data'] = self.convert_to_df(ldata)
                elif(in_format == 'list'):
                    self.module_args['data'] = self.convert_to_list(ldata)
                    
        
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
        
        Check for column related tokens 
        
        //////////////////////////////////////////////////////////////////        
        '''
                
        # indicies for tokens
        tokeniser = WhitespaceTokenizer()
        tokens_index = list(tokeniser.span_tokenize(self.command))  
        
        # we actually have column (from dataframe) data
    
        col_data = self.token_info['column'].dropna() # column names
        len_col_data = len(col_data)
        column_tokens = list(col_data.index)
        
        if(len_col_data != 0):
            
            # for each token that was found in dataframe columns            
            for token in column_tokens:
                
                # find index where it is located in input command
                matched_index_in_tokens = self.command.index(token)
                
                # all possible options we are interested
                lst_options = ['x','y','hue','col','row','target','val','splits']
                
                for option in lst_options:
                    
                    for ii,segment in enumerate(tokens_index):
                        if(matched_index_in_tokens in segment):
                            if(self.tokens[ii-1] == option):
                                self.module_args[option] = token

                        
        '''
        
        Check general plot setting tokens
        
        //////////////////////////////////////////////////////////////////
        
        '''
        
        for ii,token in enumerate(self.tokens):
            if(self.tokens[self.tokens.index(token)-1] == 'col_wrap'):
                self.module_args['col_wrap'] = token
            if(self.tokens[self.tokens.index(token)-1] == 'kind'):
                self.module_args['kind'] = token                                       
                    
        
        '''
        
        USE NER TO SORT MODULE ARGS
        
        //////////////////////////////////////////////////////////////////
        '''
        
        # only if input is a dataframe
        
        if(in_format == 'pd.DataFrame'):       
            
            request_split = self.token_split
            token_split_id = self.token_split_id     
        
            unique_nerid = token_split_id.copy()
            key_token = []
            for lst_tokens in unique_nerid:
                key_token.append([i for i in lst_tokens if i != 4])
            
            request_split = self.token_split
            token_split_id = self.token_split_id 

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
                bigram_tokens = self.n_grams(lst,2)
                trigram_tokens = self.n_grams(lst,3)          
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
        
        find general tokens 
        
        '''
        
        # lets find some general tokens 
        # uses token taging
        
        for ii,token in enumerate(self.tokens):
            
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
                          'splitter','nan_mode','bootstrap_type','l2_leaf_reg']
            
            for gtoken in lst_gtokens:
                if(self.tokens[self.tokens.index(token)-1] == gtoken):
                    self.module_args[gtoken] = token
                    
    '''
    
    Tokenisers    
    
    '''
                
    # tokenisers, return list of tokens          
                
    @staticmethod
    def nltk_tokeniser(text):
        return word_tokenize(text)
        
    @staticmethod
    def nltk_wtokeniser(text):
        return WhitespaceTokenizer().tokenize(text)
        
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

    # set self.tokens, self.bigram_tokens, self.trigram_tokens
    # set self.token_info dataframe

    def tokenise_request(self):

        # tokenise input, unigram. bigram and trigram
        self.tokens = self.nltk_wtokeniser(self.command)
        self.bigram_tokens = self.n_grams(self.tokens,2)
        self.trigram_tokens = self.n_grams(self.tokens,3)

        uni = pd.Series(self.tokens).to_frame()
        bi = pd.Series(self.bigram_tokens).to_frame()
        tri = pd.Series(self.trigram_tokens).to_frame()

        uni['type'] = 'uni'
        bi['type'] = 'bi'
        tri['type'] = 'tri'

        self.token_info = pd.concat([uni,bi,tri],axis=0)      
        self.token_info.columns = ['token','type']
        self.token_info.index = self.token_info['token']
        del self.token_info['token']
        
    def do(self,command:str,args:dict):
        
        '''
        
        Module argument
        
        '''
       
        # user input command
        self.command = command
        
        # Initialise arguments dictionary
       
        self.module_args = {'pred_task': None, 'data': None,'subset': None,'splits':None,
                            'features': None, 'target' : None,
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
            
        # tokenise input request
        self.tokenise_request()
        
        # ner splitting of request
        self.ner_split()
        
        # determine task_pred, module_name
        self.do_predict() # task_pred, module_name
        
        # logical tasks
        # sort_module_args requires function prediction
        
        self.check_data()
        self.sort_module_args()
             
        '''
        
        iterative process
        
        '''
        
        # activate relevant class & pass arguments
        nlpi.iter += 1
        
        self.module_args['pred_task'] = self.task_name
        
        nlpi.memory_name.append(self.task_name)  
        nlpi.memory_stack.append(self.module.mod_summary.loc[nlpi.memory_name[nlpi.iter]] )
        nlpi.memory_info = pd.concat(self.memory_stack,axis=1) # stack task information order
        
        # activate function

        self.module.functions[self.module_name].sel(self.module_args)
        
        # if not data has been added
        # initialise output data (overwritten in module.function
        
        if(len(nlpi.memory_output) == nlpi.iter+1):
            pass
        else:
            nlpi.memory_output.append(None) 
            