from mllibs.nlpm import nlpm
import numpy as np
import pandas as pd
import random
import json
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
from collections import Counter
import itertools
import difflib
import textwrap

from mllibs.ner_activecolumn import ac_extraction

'''
##############################################################################



                            [ INTERPRETER CLASS ] 



##############################################################################
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
      nlpi.silent = True                    # by default don't display 
      nlpi.activate = True
      nlpi.lmodule = self.module            # class variable variation for module calls
      
      # temporary active function storage
      self.tac_id = 0
      self.tac_data = {}
                  
      # class plot parameters
      nlpi.pp = {'title':None,'template':None,'background':None,'figsize':None, 
                 'stheme':None, 'fill': True,'legend':True,'bw':1}

    '''
    ###########################################################################

                            Plotting parameters nlpi.pp

    ###########################################################################
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
        nlpi.pp = {'title':None,'template':None,'background':None,
                   'figsize':None, 'stheme':None, 'fill': True, 
                   'legend':True,'bw':1}

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
    ###########################################################################

    NER TAGGING OF INPUT REQUEST
       
    ###########################################################################
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
    ###########################################################################

    Check if token names are in data sources 
    
    ###########################################################################
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
    ###########################################################################

                        Define module_args [data,data_name]

    ###########################################################################
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
        filter_words = ['as','to']
        tokens = [x for x in ltokens if x not in filter_words]
#       tokens = ltokens
        
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
                          'subset': None,'sub_task':None,
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
                
              Group Multi Columns into Temporary Active Columns
    
        When user specifies multiple column names consecutively, the
        columns are grouped together into a single gropup using temporary
        active columns which are stored in the following:

        [tac_data] : temporary storage for active columns for the instance
                     variable session

        [tac_id] : counter for dictionary storage 
    
    #######################################################################
    '''
    
    def make_tac(self):
      
      ls = self.token_info.copy()
      
      # columns
      data = list(ls['column'].fillna(0)) 
      
      # index of all b-param tokens
      b_param_idx = list(ls[ls['ner_tags'] == 'B-PARAM'].index) # index of all b-param tokens
      
      # get side by side string indicies
      def str_sidebyside(lst):
        indices = [ii for ii in range(1, len(data)-1) if isinstance(data[ii], str) and (isinstance(data[ii-1], str) or isinstance(data[ii+1], str))]
        return indices
      
      # group neighbouring numbers 
      def group_numbers(numbers):
        groups = []
        temp_group = []
        for i in range(len(numbers)-1):
          temp_group.append(numbers[i])
          if numbers[i+1] - numbers[i] != 1:
            groups.append(temp_group)
            temp_group = []
            
        try:
          temp_group.append(numbers[-1])
          groups.append(temp_group)
          return groups
        except:
          return None
      
      numbers = str_sidebyside(data)
      grouped_numbers = group_numbers(numbers)
      
      if(grouped_numbers is not None):
      
        for group in grouped_numbers:
          
          # check that all are from same dataset
          lst_col_source = list(ls.loc[group,'column'])
          column_names = list(ls.loc[group,'token'])
          same_data_check = all(x == lst_col_source[0] for x in lst_col_source) 
          
          if(same_data_check):
            
            tac_name = f'tac_data{self.tac_id}'
            self.tac_data[tac_name] = column_names
            
            ls = ls[~ls.index.isin(group)] # remove them
            ls.loc[group[0],'token'] = f"tac_data{self.tac_id}" # needs to be unique
            ls.loc[group[0],'ner_tags'] = 'O'
            ls.loc[group[0],'column'] = lst_col_source[0]
            ls.loc[group[0],'type'] = 'uni'
            ls.loc[group[0],'ttype'] = 'str'
            ls.loc[group[0],'ac'] = True
            ls = ls.sort_index()
            ls = ls.reset_index(drop=True)
            ls['index_id'] = list(ls.index)
            self.tac_id += 1
          
      # update [self.token_info]
      self.token_info = ls
            
    ''' 
    #######################################################################
                
                          Active Column Treatment
    
        [self.get_current_ac] dictionary of ac names that have been stored
                              sets [self.ac_data]
    
    
        [self.find_ac] find the data associated with the provided ac name
        [self.ac_to_columnnames] get the column names for the proviced ac name 
    
    #######################################################################
    '''
              
    # dictionary of ac names that have been stored
    # sets [self.ac_data]
    
    def get_current_ac(self):
      
      ac_data = {}
      for data_name in nlpi.data.keys():
        if(isinstance(nlpi.data[data_name]['data'],pd.DataFrame)):
          ac_data[data_name] = list(nlpi.data[data_name]['ac'].keys())
          
      self.ac_data = ac_data
      
    # find the [data] associated with the [ac name]
      
    def find_ac(self,name):
      
      data_name = None
      for key,values in self.ac_data.items():
        if(name in values):
          data_name = key
          
      if(data_name is not None):
        return data_name
      else:
        return None
      
    def find_tac(self,name):
      try:
        return self.tac_data[name]
      except:
        return None
      
    # get the [column names] associated with the active column name
      
    def ac_to_columnnames(self,ac_name:str):
      
      data_name = self.find_ac(ac_name)
      column_names = nlpi.data[data_name]['ac'][ac_name]
      return column_names
  
    # Having [self.token_info]
    # active column names associated with data sources found in the 
    # request are stored in [self.tac_data]
    # extraction of data only only for [self.token_info]
    
    def ac_to_tac_storage(self):
      
      ls = self.token_info.copy()
      
      used_data = list(ls[~ls['data'].isna()]['token'])
      self.get_current_ac()
    
      lst_ac_names = []
      for data in used_data:
        lst_ac_names.extend(nlpi.data[data]['ac'])
      
      dct_ac_mapper = {}
      for ac_name in lst_ac_names:
        dct_ac_mapper[ac_name] = self.ac_to_columnnames(ac_name)
        
      self.tac_data.update(dct_ac_mapper)
      
    # given self.token_info, store available ac names
    # into a single reference list
      
    def store_data_ac(self):
      
      ls = self.token_info.copy()
      ls['ac'] = None
      
      # data sources in current request
      used_data = list(ls[~ls['data'].isna()]['token'])
      
      for data in used_data:
        ac_names = self.ac_data[data]
        idx_ac = list(ls[ls['token'].isin(ac_names)].index)
        ls.loc[idx_ac,'ac'] = True
        ls.loc[idx_ac,'column'] = data
      
      self.token_info = ls
      
    # search for active column names in stored [self.module_args]
    # to be called before task activation, so they can be converted 
    # to the correct column names
      
    def recall_ac_names(self):
      
      ls = self.token_info.copy()
      
      for key,value in self.module_args.items():
        
        if(type(value) == str):
          
          if(self.find_ac(value) is not None or self.find_tac(value) is not None):
            
            # try the two storage locations
            try:
              self.module_args[key] = self.ac_to_columnnames(value)
            except:
              self.module_args[key] = self.tac_data[value]
            
        if(type(value) == list):
          
          print('value')
          print(value)
          
          if(len(value) > 1):
            print('[note] multiple active columns are not supported for subsets')
          elif(len(value) == 1):
            
            # try the two storage locations
            try:
              self.module_args[key] = [self.ac_to_columnnames(value[0])]
            except:
              self.module_args[key] = self.tac_data[value[0]]
              
          else:
            print('[note] something went wrong @recall_ac_names')
            
    
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

        # Active Column Related

        #######################################################################
        '''
        
        self.make_tac()        # group together any multicolumns into temporary
                               # active columns
                                
        self.ac_to_tac_storage() # store all the data source related ac names
                                 # in [self.tac_data]
        
#       self.get_current_ac()  # store all available ac names in [self.ac_data] # repeated above
        self.store_data_ac() 
      
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
        vocab_tokens = df_tinfo[(df_tinfo['vocab'] == True)]
        lst_keep_tokens = vocab_tokens['index_id']

        if(nlpi.silent is False):
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
        # function used to compare strings and returns the index
        # of tokens missing (uses .split() tokenisation)
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
        ######################################################################
        
                                [1] DATA TOKEN PARSING

          After active column data is stored, next we search for data tokens
        
        ######################################################################
        '''
      
        ls = df_tinfo.copy()
      
        '''
        
        [LABEL] Step 1 : label token_info to include [-data] tokens
        
        '''
      
        def add_datatoken_totokeninfo(ls:pd.DataFrame):
          
          # indicies at which column data is available
          data_idx = ls[~ls['data'].isna()].index.tolist()
          
          if(len(data_idx)>0):
          
            for data_row_idx in data_idx:
              
              # add new row to multicolumn dataframe at index [data_row_idx]
              new_row = pd.DataFrame([[None] * len(ls.columns)], index=[data_row_idx], columns=ls.columns) 
              new_row['token'] = '-data'
              new_row['type'] = 'uni'
              new_row['ner_tags'] = 'O'
              
              # merge the dataframe
              ls = pd.concat([ls.iloc[:data_row_idx], new_row, ls.iloc[data_row_idx:]]) 
              ls = ls.reset_index(drop=True)
              ls['index_id'] = ls.index.tolist()
              
          return ls
          
      
        '''
        
        [STORE] Step 2 : Store the [-data] value name & remove it
        
        '''
              
        # identify [-data] and store its values (next token)
        def store_data_filter_name(input_string:str):
          
          tokens = input_string.split()
          parameters = {}
          i = 0
          while i < len(tokens):
            if tokens[i].startswith("-data"):
              parameter_name = tokens[i][1:]
              if i + 1 < len(tokens):
                if parameter_name in parameters:
                  if not isinstance(parameters[parameter_name], list):
                    parameters[parameter_name] = [parameters[parameter_name]]
                  parameters[parameter_name].append(tokens[i + 1])
                else:
                  parameters[parameter_name] = tokens[i + 1]
                del tokens[i+1]
              else:
                parameters[parameter_name] = None
              i += 1
            else:
              i += 1
          return ' '.join(tokens), parameters
        
        # ls2 (token_info) + [-data] tokens inserted
        ls = add_datatoken_totokeninfo(ls)
        input_request = " ".join(list(ls['token']))
        
        result, data_parameters = store_data_filter_name(input_request)
        remove_idx = string_diff_index(input_request,result)
        ls = ls.drop(remove_idx) # update ls (exclude data names)
        ls = ls.reset_index(drop=True)
        ls['index_id'] = list(ls.index)
        
#       print("Tokenised String:", result) # result (filtered data names)
#       print("Parameter Values:", parameter_values) # stored [-data]
        
        # store the data in [module_args]
        
        try:
          print('[note] data sources have been found')
          print(data_parameters)
          self.module_args['data'] = nlpi.data[data_parameters['data']]['data']
          self.module_args['data_name'] = data_parameters['data']
        except:
          print('[note] no data source specified')
      
        '''
        ######################################################################
        
                            [2] PARAMETER TOKEN PARSING

        [1] label_params_names : add [~] to PARAM tokens (token_info adjustment)
        
        ######################################################################
        '''
      
        # add [~] labels to param tokens (not modifying the dataframe size)
        
        '''
        
                     [ 1. add [~] labels to param tokens ]
                       (not modifying the dataframe size)
        
            # eg. [~x] column [~y] ...
        
        '''
        
        def label_params_names(ls:pd.DataFrame):
          ls = ls.copy()
          # indicies at which column data is available
          ner_param_idx = ls[ls['ner_tags'].isin(['B-PARAM','I-PARAM'])].index.tolist() 
          ls.loc[ner_param_idx,'token'] = "~" + ls['token']
          return ls
        
        '''
        
                           [ 2. Add Parameter Labels ]
        
            # add [-column] tokens to token_info dataframe
            # add [-value] tokens 
            # add [-string] tokens 
        
        '''
        
        def label_params(ls:pd.DataFrame):
          
          self.ls = ls
          
          ls = ls.copy()
          param_idx = ls[ls['ner_tags'].isin(['B-PARAM','I-PARAM'])].index.tolist()
          param_type = ls.loc[param_idx,'ttype']
          
          # if PARAM is present only!
          if(len(param_idx) > 0):
            
            print('[note] parameters found!')
            col_idx = ls[~ls['column'].isna()].index.tolist()    
            ls['value_token'] = ls['token'].str.contains(r'^[-+]?[0-9]*\.?[0-9]+$')
            val_idx = ls[ls['value_token']].index.tolist()
            ls['str_param'] = ls['token'].shift(1).isin(['~mec','~dtype','~barmode'])
            str_idx = ls[ls['str_param']].index.tolist()
            
            new_row_col = [None] * len(ls.columns) # Create a new row with NaN values
            new_row_col[0] = '-column'
            new_row_val = [None] * len(ls.columns) # Create a new row with NaN values
            new_row_val[0] = '-value'
            new_row_str = [None] * len(ls.columns) # Create a new row with NaN values
            new_row_str[0] = '-string'
            new_rows = []     # Create a list to hold the new dataframe rows
            
            # Iterate through the dataframe and add the new row after each row that contains ~ in the first column
            for index, row in ls.iterrows():
                new_rows.append(row.tolist())
                if row[0].startswith('~') and index+1 in col_idx:
                    new_rows.append(new_row_col)
                if row[0].startswith('~') and index+1 in val_idx:
                    new_rows.append(new_row_val)
                if row[0].startswith('~') and index+1 in str_idx:
                    new_rows.append(new_row_str)
                  
            # Create a new dataframe from the list of rows
            ls = pd.DataFrame(new_rows, columns=ls.columns)
            
          else:
            print('[note] no parameters found!')
            
          return ls

  
        '''
        
        Parsing of [-column] [-values] [-string]
        
        '''
    
        def ner_column_parsing(request:str):
          
          # Remove "and" between two "-column" words
#         request = re.sub(r'(-column \w+) and (-column \w+)', r'\1 \2', request)
          
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
            
            # (1) Check if the token starts with "-column"
            
            if token.startswith("-column"):
              
              # Find the nearest token containing "~" to the left
              for j in range(i-1, -1, -1):
                if "~" in tokens[j]:
                  filter_idx.append([i for i in range(j+2,i+2)])
                  # Store the next token after "-column" in a list
                  column_value = param_dict.get(tokens[j], [])
                  column_value.append(tokens[i+1])
                  param_dict[tokens[j]] = column_value
                  break
                
            # (2) Check if the token starts with "-value"
                
            elif(token.startswith("-value")):
              
              # Find the nearest token containing "~" to the left
              for j in range(i-1, -1, -1):
                if "~" in tokens[j]:
                  filter_idx.append([i for i in range(j+2,i+2)])
                  # Store the next token after "-column" in a list
                  column_value = param_dict.get(tokens[j], [])
                  column_value.append(tokens[i+1])
                  param_dict[tokens[j]] = column_value
                  break
                
            # (3) Check if the token starts with "-string"
                
            elif(token.startswith("-string")):
              
              # Find the nearest token containing "~" to the left
              for j in range(i-1, -1, -1):
                if "~" in tokens[j]:
                  filter_idx.append([i for i in range(j+2,i+2)])
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
      
        '''
        [2.1] Create labels for PARAMETER & store in [token_info]
        '''
      
        # ls has been updated
      
        # new tokens are added to [token_info] is modified 
        pls = label_params_names(ls) # label PARAMS tokens add [~]
        ls2 = label_params(pls)      # label PARAMS ([-column],[-value],[-string])
              
        '''
        [2.2] Extract Parameter Values & Filter names & values
        '''
      
        # activate only if [~PARAM] is found in input request
        if not(ls2['token'].tolist() == pls['token'].tolist()):
          
          param_dict,result = ner_column_parsing(" ".join(ls2['token']))
          remove_idx = string_diff_index(" ".join(ls2['token']),result)
          ls2 = ls2.drop(remove_idx) # update ls
          ls2 = ls2.reset_index(drop=True) # reset index
          
          # update param_dict (change string to int/float if needed)
          for key, value in param_dict.items():
            if isinstance(value, list):
              param_dict[key] = [float(x) if '.' in x else int(x) if x.isdigit() else x for x in value]
            else:
              if '.' in value:
                param_dict[key] = float(value)
              else:
                param_dict[key] = int(value) if value.isdigit() else value

          print('[note] setting module_args parameters')
          
          '''
          
          Convert Set Active Column Names (if exist)
          
          '''
          
          for key,value in param_dict.items():
            
            try:
              param_dict[key] = self.tac_data[value]
            except:
              pass

          self.module_args.update(param_dict)
          print(param_dict)
      
        '''
        ######################################################################
        
                               [3] SUBSET TOKEN PARSING

          Having filtered all the [~] tokens, the next step is to check
          for remaining subset cases, ie. when a column is referenced without
          any parameter assignment
        
        ######################################################################
        '''
          
        '''
        
        Label Subset Tokens
        
          We already checked for PARAM cases so the only remaining 
          ones are [-column] by themselves 
        
        '''
                  
        def label_subset(ls:pd.DataFrame):
          
          ls = ls.copy()
          col_idx = ls[~ls['column'].isna()].index.tolist()    
          
          new_row_col = [None] * len(ls.columns) 
          new_row_col[0] = '-column'
          
          new_rows = []
          # Iterate through the dataframe and add the new row after each row that contains ~ in the first column
          for index, row in ls.iterrows():
            new_rows.append(row.tolist())
            if not row[0].startswith('~') and index+1 in col_idx:
              new_rows.append(new_row_col)
                
          # Create a new dataframe from the list of rows
          ls = pd.DataFrame(new_rows, columns=ls.columns)
                
          return ls
      
        # step 1 : group together tokens which contain "-column" and its value
      
        def merge_column_its_value(input_string:str):
          
          # Tokenize the input string
          token_list = input_string.split()
      
          grouped_tokens = []
          current_group = []
          for ii,token in enumerate(token_list):
            if token == '-column' and token_list[ii-1][0] != '~':
              current_group.append(token)
            else:
              if current_group:
                current_group.append(token)
                grouped_tokens.append(current_group)
                current_group = []
              else:
                grouped_tokens.append([token])
                
          nested_list = grouped_tokens
      
          return nested_list
        
        # step 2 : Find and merge the lists that contain "-column" within a specified window
        
        def merge_near_column_param(nested_list:list):
          
          merged_list = []
          i = 0
          
          while i < len(nested_list):
            if i < len(nested_list) - 2 and ("-column" in nested_list[i] and len(nested_list[i]) == 2) and ("-column" in nested_list[i + 2] and len(nested_list[i + 2]) == 2):
              merged_list.append(nested_list[i] + nested_list[i + 1] + nested_list[i + 2])
              i += 3
            else:
              merged_list.append(nested_list[i])
              i += 1
              
          return merged_list

        
        '''
        
        Store the most common token to key & set its values
        
        '''
        
        def store_most_common_todict(list:list):
          
          try:
            # nested list case
            unnested_list = [sublist[0] if len(sublist) == 1 else sublist for sublist in list]
            final_list = [item for sublist in unnested_list for item in (sublist if isinstance(sublist, list) else [sublist])]
          except:
            # just list
            final_list = list
            
          # Find the most common token and its next token
          token_counts = Counter(final_list)
          most_common_token = token_counts.most_common(1)[0][0]
          next_tokens = [final_list[i+1] for i in range(len(final_list)-1) if final_list[i] == most_common_token]
          
          # Store the results in a dictionary
          results = {most_common_token: next_tokens if len(next_tokens) > 1 else next_tokens[0]}
          return results
        
        '''
        
        Remove parameter values from input string 
        
          [note] called after the relevant data has been extracted
        
        '''
        
        def remove_column_parameter_values(input_string:str):
          
          # Define the pattern for tokenized values
          pattern = r'(-column\s+)\w+'
          
          # Replace the words after "-column" with an empty string
          processed_string = re.sub(pattern, r'\1', input_string)
          processed_string = re.sub(r'\s{2,}', ' ', processed_string) # Remove extra spaces
          
          return processed_string
        
  
        # label subset tokens adding [-column] to non parameter tokens
        ls3 = label_subset(ls2)
        
        '''
        
        Extract [subset] token data
        
        '''
      
        def map_values(lst, dct):

          if isinstance(lst, list):
            return [map_values(item, dct) for item in lst]
          else:
            return dct.get(lst, lst)
      
        if not(ls3['token'].tolist() == ls2['token'].tolist()):
          
          input_string = " ".join(ls3['token'])
          
          '''
        
          eg.
          ...
          ['plot'],
          ['of'],
          ['-column', 'X']]
        
          '''
          
          # merge -column & its value into a list
          nested_list = merge_column_its_value(input_string) 
          
          '''
        
          if two sets of [-subset] are in close proximity, merge them
          
          eg.
          ...
          ['plot'],
          ['of'],
          ['-column', 'X','-column', 'Y']]
          '''
          
          # group neighbouring -column into one list
          merged_list = merge_near_column_param(nested_list) 
          
          # group together and create dictionaries of column parameters
          
          list_of_dicts = []
          for lst in merged_list:
            if('-column' in lst and len(lst) != 1):
              list_of_dicts.append(store_most_common_todict(lst))
              
          merged_list = []
          for d in list_of_dicts:
            for key in d:
              if key in merged_list and d[key] == merged_list[key]:
                merged_list.append(d[key])
              else:
                merged_list.append(d[key])
          
          # standardise merged_list
          
          if isinstance(merged_list[0], list):
            merged_list = [item for sublist in merged_list for item in sublist]
          else:
            merged_list = merged_list
          
          try:
            merged_list = map_values(merged_list,self.tac_data)
          except:
            pass
                    
          # create parameter dictionary for 
          subset_param = {'column':merged_list}
          
          print('extracted [subset] parameters')
          print(subset_param)
                      
          # update [module_args]
          self.module_args.update(subset_param)
          
        # remove parameters, resultant string
        
        # ls3 added [-column] tags
        result = remove_column_parameter_values(" ".join(ls3['token']))
        
        # remove tokens (nope!)
        remove_idx = string_diff_index(" ".join(ls3['token']),result)
        ls3 = ls3.drop(remove_idx) # update ls (exclude data names)
          
        # update
        df_tinfo = ls3
      
        '''
        ######################################################################
        
                            remove [token_remove] tokens
        
        ######################################################################
        '''
        
#       # required information
#       token_id = list(df_tinfo['token'])
#       ner_id = list(df_tinfo['ner_tags'].fillna('O'))
#       
#       # remove [token_remove] tokens
#       def remove_tokens(tokens:list,ner_id:list):
#         result = [tokens[i] for i in range(len(tokens)) if ner_id[i].lower() not in ['b-token_remove', 'i-token_remove']]
#         return " ".join(result)
#     
#       filtered_request = remove_tokens(token_id,ner_id)
#       removed_idx = string_diff_index(" ".join(token_id),filtered_request)
#       
#       # update 
#       df_tinfo = df_tinfo.drop(removed_idx)
#       df_tinfo = df_tinfo.reset_index(drop=True)
  
        '''
        ######################################################################
        
                                 Filtered Request
        
        ######################################################################
        '''
  
        filtered = " ".join(list(df_tinfo['token']))
        
        if(nlpi.silent is False):
          print('\n[note] filtered request:')
          print(filtered)
      
        if(nlpi.silent is False):
          print('\n##################################################################\n')

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
            try:
              self.module_args['sub_task'] = self.module.sub_models[self.task_name].predict([filtered])
            except:
              pass

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