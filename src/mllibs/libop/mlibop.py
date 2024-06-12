
from mllibs.nlpi import nlpi
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json

'''

mllibs related operations

'''

class libop_general(nlpi):
    
    def __init__(self):
        self.name = 'libop'  

        path = pkg_resources.resource_filename('mllibs','/libop/mlibop.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)
        
    # select activation function
    def sel(self,args:dict):
                
        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name'] # {'name':'type'}
        
        if(select == 'mlibop_sdata'):
            self.stored_data(args)
        if(select == 'mlibop_functions'):
            self.stored_functions(args)
        if(select == 'mlibop_convertlisttodf'):
            self.convert_list_to_df(args)
        if(select == 'set_model_target'):
            self.set_model_target(args)


    '''

    Activation Functions

    '''

    # show the dataframe
    def stored_data(self,args:dict):

        print('[note] currently stored data: ')
        data_keys = list(nlpi.data.keys())
        print(data_keys)


    # show activation function summary dataframe
    def stored_functions(self,args:dict):  
        
        module_summary = nlpi.lmodule.mod_summary
        display(module_summary.head())
        print("[note] data stored in nlpi.memory_output; call .glr()['data']")
        nlpi.memory_output.append({'data':module_summary})

    # convert stored lists into dataframe
    def convert_list_to_df(self,args:dict):
 
        lst_data = []
        for name in args['data']['list']:
            ldata = nlpi.data[name]['data']
            data = pd.DataFrame(ldata,columns=['data'])
            data['sample'] = name
            data['ids'] = range(len(data))
            lst_data.append(data)

        combined = pd.concat(lst_data)
        combined = combined.reset_index(drop=True)
        df_pivot = combined.pivot(index='ids',columns='sample', values='data')

        nlpi.store_data(df_pivot,args['store_as'])

    def set_model_target(self,args:dict):

        '''
        
            store column as the target variable
        
        '''

        data_names = list(self.data_name.keys()) # should be one
        data_name = data_names[0]

        nlpi.data[data_name]['target'] = args['column'][0]

