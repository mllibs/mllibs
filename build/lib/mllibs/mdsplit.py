import pandas as pd
from mllibs.nlpi import nlpi
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import random


'''

Split Data into Subsets 


'''


class make_fold(nlpi):
    
    # called in nlpm
    def __init__(self,nlp_config):
        self.name = 'make_folds'             
        self.nlp_config = nlp_config 

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
    # kfold_label 
    # skfold_label
    # tts_label
        
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
        
        # store split data into input data source

        nlpi.data[self.data_name[0]]['splits'][f'kfold_{nlpi.iter}'] = kf
   
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
   
'''

Corpus

'''   

corpus_makefold = OrderedDict({"kfold_label":['create kfold',
                                              'create kfolds',
                                              'make kfold',
                                              'create kfold labels',
                                              'create subset folds',
                                              'make subset fold',
                                              'label kfold'],
                                      
                                "skfold_label": ['stratified kfold',
                                                 'stratified kfolds',
                                                 'create stratified kfold',
                                                 'make stratified kfold',
                                                 'generate stratified kfold',
                                                 'label statified kfold'],
                                            
                                'tts_label': ['train test split label',
                                              'train test split labels',
                                              'train test splitting labels',
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
                                'action':'create subset',
                                'topic':'subset generation',
                                'subtopic':'kfold cross validation',
                                'input_format':'pd.DataFrame',
                                'description':"K-fold cross-validation is a technique used in machine learning to evaluate the performance of a model. It involves dividing the dataset into k equal-sized subsets, or folds. The model is then trained on k-1 folds and tested on the remaining fold. This process is repeated k times, with each fold being used as the test set once. The results are averaged across the k iterations to provide an estimate of the model's performance. K-fold cross-validation helps to reduce the risk of overfitting and provides a more accurate estimate of the model's generalization performance. It is commonly used in machine learning to tune hyperparameters, select models, and compare different algorithms.",
                                'arg_compat':'splits shuffle rs'},
                            
                'skfold_label': {'module':'make_folds',
                                'action':'create subset',
                                'topic':'subset generation',
                                'subtopic':'stratified kfold cross validation',
                                'input_format':'pd.DataFrame',
                                'description':"Stratified k-fold cross-validation is a variation of k-fold cross-validation that ensures that each fold is representative of the overall distribution of the target variable. This is particularly useful when dealing with imbalanced datasets, where one class may be significantly underrepresented. In stratified k-fold cross-validation, the dataset is divided into k folds, but the division is done in such a way that each fold contains approximately the same proportion of samples from each class as the original dataset. This ensures that each fold is representative of the overall distribution of the target variable, and reduces the risk of bias in the evaluation of the model's performance. Stratified k-fold cross-validation is commonly used in classification tasks where the goal is to predict the class label of a sample based on its features.",
                                'arg_compat':'splits shuffle rs'
                                },

                'tts_label': {'module':'make_folds',
                              'action':'create subset',
                              'topic':'subset generation',
                              'subtopic':'train test split',
                              'input_format':'pd.DataFrame',
                              'description':"Train test splitting is a technique used in machine learning to evaluate the performance of a model. It involves dividing the available dataset into two subsets: the training set and the testing set. The training set is used to train the model, while the testing set is used to evaluate its performance. The idea behind train test splitting is to assess how well the model generalizes to new, unseen data. By evaluating the model on a separate testing set, we can get an estimate of its performance on new data that it has not seen before. The size of the training and testing sets can vary depending on the size of the dataset, but a common practice is to use 70-80 of the data for training and the remaining 20-30 for testing.",
                              'arg_compat':'test_size shuffle rs'}
                                 
                            }
                         
# configuration dictionary (passed in nlpm)
configure_makefold = {'corpus':corpus_makefold,'info':info_makefold}