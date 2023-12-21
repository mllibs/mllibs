
from mllibs.nlpi import nlpi
import plotly.express as px
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
        
        if(select == 'mlibop_sdata'):
            self.stored_data(args)
        if(select == 'mlibop_functions'):
            self.stored_functions(args)

    '''

    Activation Functions

    '''

    '''

    Show Stored Dataset Names

    '''
    # (mlibop_sdata)

    def stored_data(self,args:dict):

        print('[note] currently stored data: ')
        data_keys = list(nlpi.data.keys())
        print(data_keys)

        # print('Each dataset stores ')
        # try:
        #     idx = data_keys[0]
        #     print(list(nlpi.data[idx].keys()))
        # except:
        #     pass    

    '''

    Show Activation Function Summary DataFrame

    '''
    # (mlibop_functions)

    def stored_functions(self,args:dict):  
        
        module_summary = nlpi.lmodule.mod_summary
        display(module_summary.head())
        print("[note] data stored in nlpi.memory_output; call .glr()['data']")
        nlpi.memory_output.append({'data':module_summary})
