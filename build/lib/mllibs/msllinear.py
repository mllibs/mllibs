from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge, RidgeClassifier, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error
from mllibs.nlpi import nlpi
import pandas as pd
import numpy as np


'''

LINEAR MODEL MODULE

'''

class sllinear(nlpi):
    
    def __init__(self,nlp_config):
        self.name = 'sllinear'
        self.nlp_config = nlp_config 
        self.select = None
        self.args = None
        
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
           
    # make selection  

    def sel(self,args:dict):
    
        self.select = args['pred_task']
        self.args = args    
        
        # check feature and target variable have been defined
        set_feat = False; set_target = False
        if(len(self.args['features']) > 1):
            features = self.args['data'][self.args['features']]
            set_feat = True
        if(len(self.args['target']) == 1):
            target = self.args['data'][self.args['target'][0]]
            set_target = True
        
        if(set_feat is not True and set_target is not True):
            print('features and target not set correctly')
            return 
        
        ''' select appropriate predicted method '''
        
        # linear regression
        
        if(self.select == 'fit_lr'):
            self.sklinear_lr(features,target,self.args)
        elif(self.select == 'lr_fpred'):
            self.sklinear_lr_fpred(features,target,self.args)
            
        # logistic regression
            
        elif(self.select == 'fit_lgr'):
            self.sklinear_lgr(features,target,self.args)
        elif(self.select == 'lgr_fpred'):
            self.sklinear_lgr_fpred(features,target,self.args)

        # ridge regression
            
        elif(self.select == 'fit_ridge'):
            self.sklinear_ridge(features,target,self.args)
        elif(self.select ==  'ridge_fpred'):
            self.sklinear_ridge_fpred(features,target,self.args)
            
        # ridge classification
            
        elif(self.select == 'fit_cridge'):
            self.sklinear_cridge(features,target,self.args)
        elif(self.select ==  'cridge_fpred'):
            self.sklinear_cridge_fpred(features,target,self.args)   
            
        # lasso regression
        
        elif(self.select == 'fit_lasso'):
            self.sklinear_lasso(features,target,self.args)
        elif(self.select ==  'lasso_fpred'):
            self.sklinear_lasso_fpred(features,target,self.args)
            
        # elasticnet regression
            
        elif(self.select == "fit_elastic"):
            self.sklinear_elasticnet(features,target,self.args)
        elif(self.select ==  'elastic_fpred'):
            self.sklinear_elastic_fpred(features,target,self.args)                 
            
        # bridge regression
            
        elif(self.select == "fit_bridge"):
            self.sklinear_bridge(features,target,self.args)
        elif(self.select ==  'bridge_fpred'):
            self.sklinear_bridge_fpred(features,target,self.args)
            
            
    # fit linear regression model
            
    def sklinear_lr(self,features,target,args:dict):

        model = LinearRegression()
        model.fit(features,target)
        nlpi.memory_output.append({'data':features,
                                   'target':target,
                                   'model':model}) 
        
    # fit logistic regression model
        
    def sklinear_lgr(self,features,target,args:dict):
            
        pre = {'const':1.0}
        model = LogisticRegression(c=self.sfp(args,pre,'const'))
        model.fit(features,target)
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model}) 
        
    # fit ridge regression model (w/ regularisation)
        
    def sklinear_ridge(self,features,target,args:dict):

        pre = {'const':1.0}
        model = Ridge(alpha=self.sfp(args,pre,'const'))
        model.fit(features,target)
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model}) 
        
    # fit lasso regression model (w/ regularisation)    
        
    def sklinear_lasso(self,features,target,args:dict):

        pre = {'const':1.0}
        model = Lasso(alpha=self.sfp(args,pre,'const'))
        model.fit(features,target)
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model}) 
        
    def sklinear_elasticnet(self,features,target,args:dict):

        pre = {'const':1.0,'l1_ratio':0.5}
        model = ElasticNet(alpha=self.sfp(args,pre,'const'),
                      l1_ratio=self.sfp(args,pre,'l1_ratio'))
        model.fit(features,target)
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model}) 

        
    def sklinear_cridge(self,features,target,args:dict):

        pre = {'const':1.0}
        model = RidgeClassifier(alpha=self.sfp(args,pre,'const'))
        model.fit(features,target)
        nlpi.memory_output.append({'feature':features,
                                   'target':target,
                                   'model':model}) 
        
        
    def sklinear_bridge(self,features,target,args:dict):

        pre = {'alpha_1':1e-6,'alpha_2':1e-6,'lambda_1':1e-6,'lambda_2':1e-6}
        model = BayesianRidge(alpha_1=self.sfp(args,pre,'alpha_1'),
                              alpha_2=self.sfp(args,pre,'alpha_2'),
                              lambda_1=self.sfp(args,pre,'lambda_1'),
                              lambda_2=self.sfp(args,pre,'lambda_2')
                             )
        model.fit(features,target)
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model}) 
       
    # fit and predict linear regression model
    
    def sklinear_lr_fpred(self,features,target,args:dict):
        
        model = LinearRegression()
        model.fit(features,target)
        y_pred = model.predict(features)
        mse = mean_squared_error(target,y_pred,squared=False)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'rmse':mse}) 
        
    # fit and predict logistic regression model
    
    def sklinear_lgr_fpred(self,features,target,args:dict):
        
        model = LogisticRegression()
        model.fit(features,target)
        y_pred = model.predict(features)
        report = classification_report(target,y_pred)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'report':report}) 
        
    # fit and predict ridge regression
        
    def sklinear_ridge_fpred(self,features,target,args:dict):
        
        pre = {'const':1.0}
        model = Ridge(alpha=self.sfp(args,pre,'const'))
        model.fit(features,target)
        y_pred = model.predict(features)
        mse = mean_squared_error(target,y_pred,squared=False)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'rmse':mse}) 
        
    # fit and predict lasso regression model
    
    def sklinear_lasso_fpred(self,features,target,args:dict):
        
        pre = {'const':1.0}
        model = Lasso(alpha=self.sfp(args,pre,'const'))
        model.fit(features,target)
        y_pred = model.predict(features)
        mse = mean_squared_error(target,y_pred,squared=False)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'rmse':mse}) 
        
    # fit and predict elastic regression
        
    def sklinear_elastic_fpred(self,features,target,args:dict):
        
        pre = {'const':1.0,'l1_ratio':0.5}
        model = ElasticNet(alpha=self.sfp(args,pre,'const'),
                           l1_ratio=self.sfp(args,pre,'l1_ratio'))
        model.fit(features,target)
        y_pred = model.predict(features)
        mse = mean_squared_error(target,y_pred,squared=False)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'rmse':mse
                                }) 
        
    # fit and predict ridge classification model
    
    def sklinear_cridge_fpred(self,features,target,args:dict):
        
        pre = {'const':1.0}
        model = RidgeClassifier(alpha=self.sfp(args,pre,'const'))
        model.fit(features,target)
        y_pred = model.predict(features)
        report = classification_report(target,y_pred)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'report':report}) 
        
    # fit and predict bridge regression
        
    def sklinear_bridge_fpred(self,features,target,args:dict):
        
        pre = {'alpha_1':1e-6,'alpha_2':1e-6,'lambda_1':1e-6,'lambda_2':1e-6}
        model = BayesianRidge(alpha_1=self.sfp(args,pre,'alpha_1'),
                              alpha_2=self.sfp(args,pre,'alpha_2'),
                              lambda_1=self.sfp(args,pre,'lambda_1'),
                              lambda_2=self.sfp(args,pre,'lambda_2')
                             )
        
        model.fit(features,target)
        y_pred = model.predict(features)
        mse = mean_squared_error(target,y_pred,squared=False)
        
        nlpi.memory_output.append({'features':features,
                                   'target':target,
                                   'model':model,
                                   'y_pred':y_pred,
                                   'rmse':mse
                                }) 
        
          
dict_sllinear = {'fit_lr':['create a linear regression model',
                           'create LinearRegression model',
                           'create linear regression model',
                           'create linear model',
                           'create LinearRegression',
                           'make linear regression model',
                           'make linear regression',
                           'train linear regression model',
                           'train LinearRegression'],
                    
                 'fit_lgr': ['create a logistic regression model',
                             'create logistic regression classification',
                             'train LogisticRegression',
                             'train  logistic classification model',
                             'train logistic regression model',
                             'train logistic linear model'],
                 
                 'fit_ridge': ['create a ridge regression model',
                              'create ridge regression model',
                              'train ridge regression',
                              'train ridge regression model',
                              'train ridge linear model',
                              'train linear ridge',
                              'fit ridge regressor',
                              'fit ridge regression model'
                              ],
                 
                 'fit_cridge': ['create a ridge classification model',
                               'create ridge classifier mode',
                               'train ridge classifier',
                               'train ridge classification model',
                               'train linear ridge classification model',
                               'train linear ridge classifier',
                               'create ridge classifier',
                               'create linear ridge classifier model',
                               'create ridge classifier model',
                               'fit ridge classifier',
                               'fit rige classification model'],
                 
                 'fit_lasso': ['create a lasso regression model',
                              'create lasso regression model',
                              'train lasso regression',
                              'train lasso regression model',
                              'train lasso linear model',
                              'train linear lasso',
                              'fit lasso regressor',
                              'fit lasso regression model',
                               'fit lasso model',
                               'fit lasso',
                               'fit lasso regression'
                              ],
                 
                 'fit_elastic': ['create an elasticnet regression model',
                              'create elasticnet regression model',
                              'train elasticnet regression',
                              'train elasticnet regression model',
                              'train elasticnet linear model',
                              'train linear elasticnet',
                              'fit elasticnet regressor',
                              'fit elasticnet regression model',
                               'fit elasticnet model',
                               'fit elasticnet',
                               'fit elasticnet regression'
                              ],
                 
                 'fit_bridge': ['create an bayesian linear regression model',
                              'create bayesian regression model',
                              'train bayesian regression',
                              'train bayesian regression model',
                              'train bayesian linear model',
                              'train linear bayesian',
                              'fit bayesian regressor',
                              'fit bayesian regression model',
                               'fit bayesian model',
                               'fit bayesian',
                               'fit bayesian regression'
                              ],

                'lr_fpred' : ['fit and predict linear regression model',
                             'create and predict linear regression model',
                             'fit_predict linear regression model',
                             'fit_predict LinearRegression'],
                 
                 'lgr_fpred' : ['fit and predict logistic regression model',
                                  'fit_predict logistic regression model',
                                  'fit_predict LogisticRegression',
                                  'create logistic regression model'],
                 
                'ridge_fpred' : ['fit and predict ridge regression model',
                                 'fit and predict linear ridge regression model',
                                 'create and predict ridge regression model',
                                 'fit_predict ridge regression model',
                                 'fit_predict ridge regressor'],
                 
                'lasso_fpred' : ['fit and predict lasso regression model',
                                 'fit and predict linear lasso regression model',
                                 'create and predict lasso regression model',
                                 'fit_predict lasso regression model',
                                 'fit_predict lasso regression model', 
                                 'fit_predict lasso regressor'],
                 
                'elastic_fpred' : ['fit and predict elasticnet regression model',
                                 'fit and predict linear elasticnet regression model',
                                 'create and predict elasticnet regression model',
                                 'fit_predict elasticnet regression model',
                                 'fit_predict elasticnet regression model', 
                                 'fit_predict elasticnet regressor'],
                 
                'cridge_fpred' : ['fit and predict ridge classification model',
                                 'fit and predict linear ridge classification model',
                                 'create and predict ridge classification model',
                                 'fit_predict ridge classification model',
                                 'fit_predict ridge classification model', 
                                 'fit_predict ridge classifier'],
                 
                'bridge_fpred' : ['fit and predict bayesian ridge regression model',
                                 'fit and predict linear bayesian ridge regression model',
                                 'create and predict bayesian ridge regression model',
                                 'fit_predict bayesian ridge regression model',
                                 'fit_predict bayesian ridge regression model', 
                                 'fit_predict bayesian ridge regressor'],
                
                }


info_sllinear = {'fit_lr':{'module':'sllinear',
                            'action':'train model',
                            'topic':'linear regression',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':'Using the sklearn module, create a linear regression (LinearRegression) regression model',
                      'token_compat':'data features target'},

                    'fit_lgr':{'module':'sllinear',
                                'action':'train model',
                                'topic':'linear classification',
                                'subtopic':'model training',
                                'input_format':'pd.DataFrame',
                               'description':'Using the sklearn module, create a logistic regression (LogisticRegression) classification model',
                      'token_compat':'data features target',
                             'arg_compat':'const'},

                    'fit_ridge':{'module':'sllinear',
                                'action':'train model',
                                'topic':'linear regression',
                                'subtopic':'model training',
                                'input_format':'pd.DataFrame',
                                'description':'Linear least squares with l2 regularization. This model solves a regression model where the loss function is the linear least squares function and regularization is given by the l2-norm. Also known as Ridge Regression or Tikhonov regularization',
                      'token_compat':'data features target',
                      'arg_compat':'const'},
                 
                    'fit_cridge':{'module':'sllinear',
                                'action':'train model',
                                'topic':'linear classification',
                                'subtopic':'model training',
                                'input_format':'pd.DataFrame',
                                'description':'Classifier using Ridge regression. This classifier first converts the target values into {-1, 1} and then treats the problem as a regression task',
                      'token_compat':'data features target',
                      'arg_compat':'const l1_ratio',},
                 
                    'fit_lasso':{'module':'sllinear',
                                'action':'train model',
                                'topic':'linear regression',
                                'subtopic':'model training',
                                'input_format':'pd.DataFrame',
                                'description':'Linear Model trained with L1 prior as regularizer (aka the Lasso). Technically the Lasso model is optimizing the same objective function as the Elastic Net with l1_ratio=1.0 (no L2 penalty)',
                      'token_compat':'data features target',
                      'arg_compat':'const'},

                 
                    'fit_elastic':{'module':'sllinear',
                                'action':'train model',
                                'topic':'linear regression',
                                'subtopic':'model training',
                                'input_format':'pd.DataFrame',
                                'description':'Linear regression with combined L1 and L2 priors as regularizer. l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha',
                      'token_compat':'data features target',
                      'arg_compat':'const l1_ratio'},
                 
                    'fit_bridge':{'module':'sllinear',
                                 'action':'train model',
                                 'topic':'linear regression',
                                 'subtopic':'model training',
                                 'input_format':'pd.DataFrame',
                                 'description':'Bayesian Ridge is a regression algorithm that uses Bayesian inference to estimate the parameters of a linear regression model. It is a regularized version of linear regression that adds a penalty term to the likelihood function, which helps to prevent overfitting and improve the generalization performance of the model. In Bayesian Ridge, the prior distribution over the model parameters is assumed to be a Gaussian distribution with zero mean and a diagonal covariance matrix. The hyperparameters of this prior distribution are learned from the data using maximum likelihood estimation or Bayesian inference',
                      'token_compat':'data features target',
                      'arg_compat':'alpha_1 alpha_2 lambda_1 lambda_2',  
},
                 
                    'lr_fpred':{'module':'sllinear',
                               'action':'model predict',
                               'topic':'linear regression',
                               'subtopic':'model prediction',
                               'input_format':'pd.DataFrame',
                               'description':'Using the sklearn module, create a linear (LinearRegression) regression model'},
        
                    'lgr_fpred':{'module':'sllinear',
                              'action':'model predict',
                              'topic':'linear classification',
                              'subtopic':'model prediction',
                              'input_format':'pd.DataFrame',
                              'description':'Using the sklearn module, create a linear (LogisticRegression) regression modeland use it to make a prediction on the data it was trained on',
                      'token_compat':'data features target',
                      'arg_compat':'const'},

                    'ridge_fpred':{'module':'sllinear',
                               'action':'model predict',
                               'topic':'linear regression',
                               'subtopic':'model prediction',
                               'input_format':'pd.DataFrame',
                               'description':'Using the sklearn module, create a ridge regression (Ridge) model and use it to make a prediction on the data it was trained on',
                      'token_compat':'data features target',
                      'arg_compat':'const'},
                
                     'lasso_fpred':{'module':'sllinear',
                               'action':'model predict',
                               'topic':'linear regression',
                               'subtopic':'model prediction',
                               'input_format':'pd.DataFrame',
                               'description':'Using the sklearn module, create a linear (Lasso) regression modeland use it to make a prediction on the data it was trained on',
                      'token_compat':'data features target',
                      'arg_compat':'const'},
                      

                     'elastic_fpred':{
                               'module':'sllinear',
                               'action':'model predict',
                               'topic':'linear regression',
                               'subtopic':'model prediction',
                               'input_format':'pd.DataFrame',
                               'description':'Using the sklearn module, create a linear (ElasticNet) regression modeland use it to make a prediction on the data it was trained on',
                      'token_compat':'data features target',
                      'arg_compat':'const l1_ratio',
                      },
                 
                     'cridge_fpred':{'module':'sllinear',
                               'action':'model predict',
                               'topic':'linear classification',
                               'subtopic':'model prediction',
                               'input_format':'pd.DataFrame',
                               'description':'Using the sklearn module, create a ridge classification (RidgeClassifier) model and use it to make a prediction on the data it was trained on',
                      'token_compat':'data features target',
                      'arg_compat':'const'},                     
                
                     'bridge_fpred':{'module':'sllinear',
                               'action':'model predict',
                               'topic':'linear regression',
                               'subtopic':'model prediction',
                               'input_format':'pd.DataFrame',
                               'description':'Bayesian Ridge is a regression algorithm that uses Bayesian inference to estimate the parameters of a linear regression model. It is a regularized version of linear regression that adds a penalty term to the likelihood function, which helps to prevent overfitting and improve the generalization performance of the model. In Bayesian Ridge, the prior distribution over the model parameters is assumed to be a Gaussian distribution with zero mean and a diagonal covariance matrix. The hyperparameters of this prior distribution are learned from the data using maximum likelihood estimation or Bayesian inference',
                      'token_compat':'data features target',
                      'arg_compat':'alpha_1 alpha_2 lambda_1 lambda_2',                            
                }}
        


configure_sllinear = {'corpus':dict_sllinear,'info':info_sllinear}               