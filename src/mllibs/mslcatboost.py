
from catboost import CatBoostClassifier,CatBoostRegressor
from mllibs.common_eval import eval_base


'''

CATBOOST DEDICATED MODULE

'''

# Catboost gradient boosting library related operations
# like gradientboo

# requires parent evaluation class 
# we only need to replace class name & set model attribute/method

class slcatboost(eval_base):

    def __init__(self,nlp_config):
        self.name = 'slcatboost'
        eval_base.__init__(self,nlp_config,self.name)

    def set_model(self,args):

        # select model

        # catboost regression [fit] and [fit] [predict]

        if(self.select == 'fit_rcat' or self.select == 'rcat_fpred'):   

            self.model_type = 'reg'
            pre = {'n_estimators':None,
                   'epoch':None,
                   'lr':None,
                   'loss':None,
                   'max_depth':None,
                   'nan_mode':None,
                   'bootstrap_type':None,
                   'l2_leaf_reg':None
            }

            self.model = CatBoostRegressor(iterations=self.sfp(args,pre,'epoch'),
                                           n_estimators=self.sfp(args,pre,'n_estimators'),
                                           learning_rate=self.sfp(args,pre,'lr'),
                                           loss_function=self.sfp(args,pre,'loss'),
                                           max_depth=self.sfp(args,pre,'max_depth'),
                                           nan_mode=self.sfp(args,pre,'nan_mode'),
                                           bootstrap_type=self.sfp(args,pre,'bootstrap_type'),
                                           l2_leaf_reg = self.sfp(args,pre,'l2_leaf_reg'),
                                           silent=True
                                          )
            
        # catboost classifier [fit] and [fit] [predict]
            
        elif(self.select == 'fit_ccat' or self.select == 'ccat_fpred'):   

            self.model_type = 'class'
            pre = {'n_estimators':None,
                   'epoch':None,
                   'lr':None,
                   'loss':None,
                   'max_depth':None,
                   'nan_mode':None,
                   'bootstrap_type':None,
                   'l2_leaf_reg':None
            }

            self.model = CatBoostClassifier(iterations=self.sfp(args,pre,'epoch'),
                                           n_estimators=self.sfp(args,pre,'n_estimators'),
                                           learning_rate=self.sfp(args,pre,'lr'),
                                           loss_function=self.sfp(args,pre,'loss'),
                                           max_depth=self.sfp(args,pre,'max_depth'),
                                           nan_mode=self.sfp(args,pre,'nan_mode'),
                                           bootstrap_type=self.sfp(args,pre,'bootstrap_type'),
                                           l2_leaf_reg = self.sfp(args,pre,'l2_leaf_reg'),
                                           silent=True
                                          )


'''

Module Corpus 

'''

dict_slcatboost = { 'fit_rcat' : ['create a catboost regression model',
                                'create a catboost regressor',
                                'create catboost Regressor',
                                'create a catboost gradient boosting regression model',
                                'fit catboost regression model',
                                'fit catboost gradient boosting regression model',
                                'fit catboost gradient boosting regressor',
                                'fit regressor catboost gradient boosting model',
                                'fit CatBoostRegressor',
                                'regressor catboost tree model',
                                'regressor cat boost model',
                                'catboost regression model',
                                'train regression catboost model',
                                'train catboost regressor',
                                'train catboost regression model',
                                'train CatBoostRegressor',
                                'train catboost regressor',
                                'train CatBoost regressor'],
               
                'rcat_fpred': ['fit and predict CatBoostRegressor',
                               'fit and predict CatBoost regressor',
                              'fit and predict catboost regressor',
                              'fit and predict catboost regression model',
                              'create and predict catboost regression model',
                              'create and predict catboost regressor',
                              'craete and predict CatBoostRegressor',
                              'fit_predict CatBoostRegressor',
                              'fit_predict catboost regressor',
                              'fit_predict CatBoost regressor',
                              'fit_predict CatBoost regression model',
                              'fit_predict catboost regressor',
                              'predict catboost regressor',
                              ],

                'fit_ccat' : ['create a catboost classification model',
                             'create a catboost classifier',
                             'create catboost classifier',
                             'create a catboost gradient boosting classifier model',
                             'fit catboost classifier model',
                             'fit catboost gradient boosting classification model',
                             'fit regressor catboost gradient boosting model',
                             'fit CatBoostClassifier',
                             'classifier catboost tree model',
                             'classifier cat boost model',
                             'catboost classifiaction model',
                             'train classification catboost model',
                             'train catboost classifier',
                             'train catboost classification model',
                             'train CatBoostClassifier',
                             'train catboost classifier',
                             'train CatBoost classifier'
                             ],

                'ccat_fpred': ['fit and predict CatBoostClassifier',
                               'fit and predict CatBoost classifier',
                               'fit and predict catboost classifier',
                               'fit and predict catboost classifier model',
                               'fit and predict catboost classification model',
                               'create and predict catboost clasification model',
                               'create and predict catboost classifier',
                               'create and predict CatBoostClassifier',
                               'fit_predict CatBoostClassifier',
                               'fit_predict catboost classifier',
                               'fit_predict CatBoost classifier',
                               'fit_predict CatBoost classifier model',
                               'fit_predict catboost classifier',
                               'predict catboost classification model'
                              ],
}

info_slcatboost = {
    
                'fit_rcat':{'module':'slcatboost',
                            'action':'train model',
                            'topic':'ensemble regression',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"CatBoost uses an ensemble of decision trees as base models. These decision trees are trained iteratively using gradient boosting, where each new tree is added to the ensemble to minimize the loss function. The predictions from all the trees in the ensemble are combined to make the final prediction.",
                            'token_compat':'data features target'},

                'rcat_fpred':{'module':'slcatboost',
                            'action':'model predict',
                            'topic':'ensemble regression',
                            'subtopic':'model prediction',
                            'input_format':'pd.DataFrame',
                            'description':"CatBoost uses an ensemble of decision trees as base models. These decision trees are trained iteratively using gradient boosting, where each new tree is added to the ensemble to minimize the loss function. The predictions from all the trees in the ensemble are combined to make the final prediction.",
                            'token_compat':'data features target'},

                'fit_ccat':{'module':'slcatboost',
                            'action':'train model',
                            'topic':'ensemble classification',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"CatBoost uses an ensemble of decision trees as base models. These decision trees are trained iteratively using gradient boosting, where each new tree is added to the ensemble to minimize the loss function. The predictions from all the trees in the ensemble are combined to make the final prediction.",
                            'token_compat':'data features target'},

                'ccat_fpred':{'module':'slcatboost',
                            'action':'model predict',
                            'topic':'ensemble classification',
                            'subtopic':'model prediction',
                            'input_format':'pd.DataFrame',
                            'description':"CatBoost uses an ensemble of decision trees as base models. These decision trees are trained iteratively using gradient boosting, where each new tree is added to the ensemble to minimize the loss function. The predictions from all the trees in the ensemble are combined to make the final prediction.",
                            'token_compat':'data features target'}

}

configure_slcatboost = {'corpus':dict_slcatboost,'info':info_slcatboost}      
