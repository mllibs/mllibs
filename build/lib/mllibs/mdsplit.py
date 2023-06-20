import pandas as pd
from mllibs.nlpi import nlpi
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


'''

Split Data into Subsets 


'''


class make_fold(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'make_folds'             
        self.nlp_config = nlp_config 
        
    # called in nlpi
    def sel(self,args:dict):
        
        self.select = args['pred_task']
        self.args = args
        
        if(self.select == 'kfold_label'):
            self.kfold_label(self.args)
        elif(self.select == 'skfold_label'):
            self.skfold_label(self.args)
        elif(self.select == 'tts_label'):
            self.tts_label(self.args)
        
    # Kfold splitting
        
    def kfold_label(self,args:dict):
       
        kf = KFold(n_splits=eval(args['splits']), 
                   shuffle=eval(args['shuffle']), 
                   random_state=eval(args['rs']))
                    
        for i, (_, v_ind) in enumerate(kf.split(args['data'])):
            args['data'].loc[args['data'].index[v_ind], 'kfold'] = f"fold{i+1}"
        
        # store relevant data about operation
        nlpi.memory_output.append({'data':args['data'],
                                   'shuffle':args['shuffle'],
                                   'n_splits':args['splits'],
                                   'split':kf,
                                   'rs':args['rs']}) 
                    
    # Stratified kfold splitting             
    
    def skfold_label(self,args:dict):
        
        if(type(args['y']) is str):

            kf = StratifiedKFold(n_splits=eval(args['splits']), 
                                 shuffle=eval(args['shuffle']), 
                                 random_state=eval(args['rs']))
                        
            for i, (_, v_ind) in enumerate(kf.split(args['data'],args['data'][[args['y']]])):
                args['data'].loc[args['data'].index[v_ind], 'skfold'] = f"fold{i+1}"
                
            # store relevant data about operation
            nlpi.memory_output.append({'data':args['data'],
                                       'shuffle':args['shuffle'],
                                       'n_splits':args['splits'],
                                       'stratify':args['y'],
                                       'split':kf,
                                       'rs':args['rs']}) 
        else:
            print('specify y data token for stratification!')    
            nlpi.memory_output(None)                           
            
        
    # Train test split labeling (one df only)
        
    def tts_label(self,args:dict):
        
        train, test = train_test_split(args['data'],
                                       test_size=eval(args['test_size']),
                                       shuffle=eval(args['shuffle']),
                                       stratify=args['y'],
                                       random_state=eval(args['rs'])
                                       )
        
        train['tts'] = 'train'
        test['tts'] = 'test'
        ldf = pd.concat([train,test],axis=0)
        ldf = ldf.sort_index()
        
        # store relevant data about operation
        nlpi.memory_output.append({'data':ldf,
                                   'stratified':args['y'],
                                   'shuffle':args['shuffle'],
                                   'stratify':args['y'],
                                   'test_size':args['test_size'],
                                   'rs':args['rs']}
                                   )
   
'''


Corpus


'''   

corpus_makefold = OrderedDict({"kfold_label":['create kfold',
                                      'make kfold'
                                      'create subset folds',
                                      'make subset fold',
                                      'label kfold'],
                                      
                                "skfold_label": ['stratified kfold',
                                            'create stratified kfold',
                                            'make stratified kfold',
                                            'generate stratified kfold',
                                            'label statified kfold'],
                                            
                                'tts_label': ['train test split label',
                                             'create tts label',
                                             'make tts label',
                                             'make train test split label',
                                             'train-test-split label',
                                             'create train-test-split label',
                                             'label tts',
                                             'tts labels',
                                             'create tts labels']
                                      
                                      })


info_makefold = {'kfold_label': {'module':'make_folds',
                            'action':'action',
                            'topic':'topic',
                            'subtopic':'sub topic',
                            'input_format':'pd.DataFrame',
                            'description':'generate kfolds labels for dataframe',
                            'arg_compat':'splits shuffle rs'},
                            
                'skfold_label': {'module':'make_folds',
                            'action':'action',
                            'topic':'topic',
                            'subtopic':'sub topic',
                            'input_format':'pd.DataFrame',
                            'description':'generate stratified kfolds labels for dataframe'},

                'tts_label': {'module':'make_folds',
                            'action':'action',
                            'topic':'topic',
                            'subtopic':'sub topic',
                            'input_format':'pd.DataFrame',
                            'description':'generate train-test-split labels for dataframe'}
                            
                                 
                            }
                         
# configuration dictionary (passed in nlpm)
configure_makefold = {'corpus':corpus_makefold,'info':info_makefold}