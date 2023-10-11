import pandas as pd
from mllibs.nlpi import nlpi
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import random
from mllibs.nlpm import parse_json
import json


'''

Split Data into Subsets 

'''

class make_fold(nlpi):
    
    # called in nlpm
    def __init__(self):
        self.name = 'make_folds'  

        # read config data
        with open('src/mllibs/corpus/mdsplit.json', 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)

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
        
    # called in nlpi
    def sel(self,args:dict):
        
        # define instance parameters
        self.select = args['pred_task']
        self.args = args
        self.data_name = args['data_name']  # name of the data
        
        if(self.select == 'kfold_label'):
            self.kfold_label(self.args)
        elif(self.select == 'skfold_label'):
            self.skfold_label(self.args)
        elif(self.select == 'tts_label'):
            self.tts_label(self.args)
        
    ''' 
    
    ACTIVATION FUNCTIONS 
    
    '''
        
    def kfold_label(self,args:dict):

        pre = {'splits':3,'shuffle':True,'rs':random.randint(1,500)}
       
        kf = KFold(n_splits=self.sfp(args,pre,'n_splits'), 
                   shuffle=self.sfp(args,pre,'shuffle'), 
                   random_state=self.sfp(args,pre,'rs'))
                    
        for i, (_, v_ind) in enumerate(kf.split(args['data'])):
            args['data'].loc[args['data'].index[v_ind], 'kfold'] = f"fold{i+1}"
        
        # store relevant data about operation
        nlpi.memory_output.append({'data':args['data'],
                                   'shuffle':self.sfp(args,pre,'shuffle'),
                                   'n_splits':self.sfp(args,pre,'splits'),
                                   'split':kf,
                                   'rs':self.sfp(args,pre,'rs')})
        
        # store split data in model evaluation form
        nlpi.data[self.data_name[0]]['splits'][f'kfold_{nlpi.iter}'] = kf

        # store split data in dataframe column form
        nlpi.data[self.data_name[0]]['splits_col'][f'kfold_{nlpi.iter}'] = args['data']['kfold']

        # remove column

        args['data'].drop(['kfold'],axis=1,inplace=True)
   
    # Stratified kfold splitting             
    
    def skfold_label(self,args:dict):

        pre = {'splits':3,'shuffle':True,'rs':random.randint(1,500)}
        
        if(type(args['y']) is str):

            kf = StratifiedKFold(n_splits=self.sfp(args,pre,'n_splits'), 
                                 shuffle=self.sfp(args,pre,'shuffle'), 
                                 random_state=self.sfp(args,pre,'rs'))
                        
            for i, (_, v_ind) in enumerate(kf.split(args['data'],args['data'][[args['y']]])):
                args['data'].loc[args['data'].index[v_ind], 'skfold'] = f"fold{i+1}"
                
            # store relevant data about operation
            nlpi.memory_output.append({'data':args['data'],
                                       'shuffle':self.sfp(args,pre,'shuffle'),
                                       'n_splits':self.sfp(args,pre,'splits'),
                                       'stratify':args['y'],
                                       'split':kf,
                                       'rs':self.sfp(args,pre,'rs')}) 
            
            # store relevant data about operation
            nlpi.data[self.data_name[0]]['splits'][f'skfold_{nlpi.iter}'] = kf
            nlpi.data[self.data_name[0]]['splits_col'][f'kfold_{nlpi.iter}'] = args['data']['skfold']

            # remove column

            args['data'].drop(['skfold'],axis=1,inplace=True)
            
        else:
            print('specify y data token for stratification!')    
            nlpi.memory_output(None)                           
            
        
    # Train test split labeling (one df only)
        
    def tts_label(self,args:dict):

        # preset setting 
        pre = {'test_size':0.3,'shuffle':True,'rs':random.randint(1,500)}
        
        train, test = train_test_split(args['data'],
                                       test_size=self.sfp(args,pre,'test_size'),
                                       shuffle=self.sfp(args,pre,'shuffle'),
                                       stratify=args['y'],
                                       random_state=self.sfp(args,pre,'rs')
                                       )
        
        train['tts'] = 'train'
        test['tts'] = 'test'
        ldf = pd.concat([train,test],axis=0)
        ldf = ldf.sort_index()
        
        # store relevant data about operation
        nlpi.memory_output.append({'data':ldf,
                                   'shuffle':self.sfp(args,pre,'shuffle'),
                                   'stratify':args['y'],
                                   'test_size':self.sfp(args,pre,'test_size'),
                                   'rs':self.sfp(args,pre,'rs')}
                                )

        # store relevant data about operation in data source
        nlpi.data[self.data_name[0]]['splits'][f'tts_{nlpi.iter}'] = ldf['tts']
        nlpi.data[self.data_name[0]]['splits_col'][f'tts_{nlpi.iter}'] = ldf['tts']
