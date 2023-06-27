from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge, RidgeClassifier, Lasso, ElasticNet, BayesianRidge
from mllibs.common_eval import eval_base


'''

LINEAR MODEL MODULE

'''

# requires parent evaluation class 
# we only need to replace class name & set model attribute/method

class sllinear(eval_base):

    def __init__(self,nlp_config):
        self.name = 'sllinear'
        eval_base.__init__(self,nlp_config,self.name)


    def set_model(self,args):

        # select model (7 variations - 5 regressors / 2 classifiers)

        if(self.select == 'fit_lr' or self.select == 'lr_fpred'):   

            self.model_type = 'reg'
            self.model = LinearRegression()

        elif(self.select == 'fit_lgr' or self.select == 'lgr_fpred'):

            self.model_type = 'class'
            pre = {'const':1.0}
            self.model = LogisticRegression(c=self.sfp(args,pre,'const'))

        elif(self.select == 'fit_ridge' or self.select == 'ridge_fpred'):

            self.model_type = 'reg'
            pre = {'const':1.0}
            self.model = Ridge(alpha=self.sfp(args,pre,'const'))

        elif(self.select == 'fit_cridge' or self.select == 'cridge_fpred'):

            self.model_type = 'class'
            pre = {'const':1.0}
            self.model = RidgeClassifier(alpha=self.sfp(args,pre,'const'))

        elif(self.select == 'fit_lasso'):

            self.model_type = 'reg'
            pre = {'const':1.0}
            self.model = Lasso(alpha=self.sfp(args,pre,'const'))

        elif(self.select == 'fit_elastic' or self.select == 'elastic_fpred'):

            self.model_type = 'reg'
            pre = {'const':1.0,'l1_ratio':0.5}
            self.model = ElasticNet(alpha=self.sfp(args,pre,'const'),
                                l1_ratio=self.sfp(args,pre,'l1_ratio'))
            
        elif(self.select == 'fit_bridge' or self.select == 'bridge_fpred'):

            self.model_type = 'reg'
            pre = {'alpha_1':1e-6,'alpha_2':1e-6,'lambda_1':1e-6,'lambda_2':1e-6}
            self.model = BayesianRidge(alpha_1=self.sfp(args,pre,'alpha_1'),
                                    alpha_2=self.sfp(args,pre,'alpha_2'),
                                    lambda_1=self.sfp(args,pre,'lambda_1'),
                                    lambda_2=self.sfp(args,pre,'lambda_2')
                                    )

''' 

Corpus

'''
        
          
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