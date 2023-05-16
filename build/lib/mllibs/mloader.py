import os
import pandas as pd
from collections import OrderedDict
from mllibs.nlpi import nlpi

# class that will store task methods
class loader(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'loader'
        self.nlp_config = nlp_config 
        self.select = None
        self.data = None
        self.args = None
        self.input = '/kaggle/input/'      # pathway to kaggle inputs (kaggle only)
        self.working = '/kaggle/working/'  # pathway to kaggle working (kaggle only)
        self.cwd = '/kaggle/'              # pathway to home
    
        # examine local folder and store pathways
        self.examine_home()
     
    # store / update available data file pathways
    
    def examine_home(self):

        lst = []
        lst_data_origins = [self.input,self.working]
        for origin in lst_data_origins:            
            for dirname, _, filenames in os.walk(origin):
                for filename in filenames:
                    lst.append(os.path.join(dirname, filename))

        # stores pathway data for all files @ instantiation
        nlpi.lstcwd = lst

    # make selection  
        
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.data = args['data']
        self.args = args
               
        # select 0 - store / update available data 
        
        if(self.select == 'kaggle_folders'):
            nlpi.memory_output.append(nlpi.lstcwd)
              
        # select 1 - find csv file paths
            
        if(self.select == 'find_csv'):
            
            # update available data sources
            self.examine_home()

            lst = []
            for file_path in nlpi.lstcwd:
                if('.csv' in file_path):
                    lst.append(file_path)
                        
            nlpi.memory_output.append(lst)
            
        # 3. find csvs, read them and store them in list
            
        if(self.select == 'store_csv'):
            
            lst = []        # list which stored dataframes
            lst_names = []  # list which stores file name of dataframe
            
            # update available data sources
            self.examine_home()
            
            for file_path in nlpi.lstcwd:
                if('.csv' in file_path):
                    lst_names.append(os.path.split(file_path)[1].split('.csv')[0])
                    lst.append(pd.read_csv(file_path))
                   
            # check for multiple files, store all read csvs separately
            
            if(len(lst) > 1):     
                dict_storage = {}
                for i,df in enumerate(lst):
                    dict_storage[lst_names[i]] = df
    
                nlpi.memory_output.append(dict_storage)
                    
            else:

                nlpi.memory_output.append(lst[0])
                
                
# Commands for NLP module related tasks
corpus_loader = OrderedDict({

# store all home data 
'kaggle_folders':['store kaggle input folder contents',
                'input folder contents',
                'store input folder',
                'get input folder contents',
                'get input folder content',
                'input folder'
                'kaggle input',
                'kaggle working folder',
                'update kaggle pathways',
                'get kaggle pathways',
                'store working folder'],

# find pathways to all csv files 
'find_csv':['find csv path',
            'get csv paths',
            'find csv paths',
            'csv file paths',
            'seach csv paths',
            'search for csv',
            'csv search',
            'seach csv',
            'find csv',
           ],
  
# load all csv files, if not referenced in command
# find and store all csv files in nlpi.lstcwd
'store_csv': ['store csv',
                'store csv format',
                'load csv',
                'csv load',
                'load csv',
                'read csv']

})

# Other useful information about the task
info_loader = {'kaggle_folders':{'module':'loader',
                                 'action':'get path',
                                 'topic':'system operation',
                                 'subtopic':'find folder contents',
                                 'input_format':'None',
                                 'output_format':'list',
                                 'description':'store kaggle input folder content pathways'},
                  
                 'find_csv':{'module':'loader',
                             'action':'get path',
                             'topic':'system operation',
                             'subtopic':'find folder contents',
                             'input_format':'None',
                             'output_format':'list',
                             'description':'show input folder content pathways'},
                 
                 'store_csv': {'module':'loader',
                               'action':'store data',
                               'topic':'system operation',
                               'subtopic':'read csv data',
                               'input_format':'None',
                               'output_format':'list',
                               'description':'find and store CSV format data'},
                  
                 }

configure_loader = {'corpus':corpus_loader,'info':info_loader}