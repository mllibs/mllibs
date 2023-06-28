from sklearn.ensemble import AdaBoostRegressor,AdaBoostClassifier,RandomForestRegressor,RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from mllibs.common_eval import eval_base


'''

ENSEMBLE MODEL MODULE

'''

# sklearn linear models from [.ensemble]

# requires parent evaluation class 
# we only need to replace class name & set model attribute/method

class slensemble(eval_base):

    def __init__(self,nlp_config):
        self.name = 'slensemble'
        eval_base.__init__(self,nlp_config,self.name)


    def set_model(self,args):

        # select model

        # adaboost regression [fit] and [fit] [predict]
        if(self.select == 'fit_rada' or self.select == 'rada_fpred'):   

            self.model_type = 'reg'
            pre = {'n_estimators':50,'lr':1.0,'estimator':None,'loss':'linear'}

            self.model = AdaBoostRegressor(estimator=self.sfp(args,pre,'estimator'),
                                           n_estimators=self.sfp(args,pre,'n_estimators'),
                                           learning_rate=self.sfp(args,pre,'lr'),
                                           loss=self.sfp(args,pre,'loss')                       ,
                                          )

        # adaboost classifier [fit] and [fit] [predict]
        elif(self.select == 'fit_cada' or self.select == 'cada_fpred'):

            self.model_type = 'class'
            pre = {'n_estimators':50,'lr':1.0,'estimator':None}

            self.model = AdaBoostClassifier(estimator=self.sfp(args,pre,'estimator'),
                                  n_estimators=self.sfp(args,pre,'n_estimators'),
                                  learning_rate=self.sfp(args,pre,'lr'),
                                  )
            
        # random forest regressor [fit] and [fit] [predict]
        elif(self.select == 'fit_rrf' or self.select == 'rrf_fpred'):

            self.model_type = 'reg'
            pre = {'n_estimators':100,
                "criterion":'squared_error',
                'min_samples_leaf':1,
                'min_samples_split':2,
                'max_depth':None,
                'max_features':1.0,
                'bootstrap':True,
                'oob_score':False}

            self.model = RandomForestRegressor(
                                        n_estimators=self.sfp(args,pre,'n_estimators'),
                                        criterion=self.sfp(args,pre,'criterion'),
                                        min_samples_leaf=self.sfp(args,pre,'min_samples_leaf'),
                                        min_samples_split=self.sfp(args,pre,'min_samples_split'),
                                        max_depth=self.sfp(args,pre,'max_depth'),
                                        bootstrap=self.sfp(args,pre,'bootstrap'),
                                        oob_score=self.sfp(args,pre,'oob_score'),
                                    )
            
        # random forest regressor [fit] and [fit] [predict]
        elif(self.select == 'fit_crf' or self.select == 'crf_fpred'):

            self.model_type = 'class'
            pre = {'n_estimators':100,
                "criterion":'gini',
                'min_samples_leaf':1,
                'min_samples_split':2,
                'max_depth':None,
                'max_features':1.0,
                'bootstrap':True,
                'oob_score':False}

            self.model = RandomForestClassifier(
                                        n_estimators=self.sfp(args,pre,'n_estimators'),
                                        criterion=self.sfp(args,pre,'criterion'),
                                        min_samples_leaf=self.sfp(args,pre,'min_samples_leaf'),
                                        min_samples_split=self.sfp(args,pre,'min_samples_split'),
                                        max_depth=self.sfp(args,pre,'max_depth'),
                                        bootstrap=self.sfp(args,pre,'bootstrap'),
                                        oob_score=self.sfp(args,pre,'oob_score'),
                                    )

        elif(self.select == 'fit_rhb' or self.select == 'rhb_fpred'):

            self.model_type = 'reg'
            pre = {'loss':'squared_error',
                'lr':0.1,
                'epoch':100,
                'max_depth':None,
                'min_samples_leaf':20,
                'max_bins':255,
                'validation_fraction':0.1,
                'n_iter_no_change':10,
            }

            self.model = HistGradientBoostingRegressor(loss=self.sfp(args,pre,'loss'),
                                                learning_rate=self.sfp(args,pre,'lr'),
                                                max_iter=self.sfp(args,pre,'epoch'),
                                                max_depth=self.sfp(args,pre,'max_depth'),
                                                min_samples_leaf=self.sfp(args,pre,'min_samples_leaf'),
                                                max_bins=self.sfp(args,pre,'max_bins'),
                                                validation_fraction=self.sfp(args,pre,'validation_fraction'),
                                                n_iter_no_change=self.sfp(args,pre,'n_iter_no_change')

                                  )

        elif(self.select == 'fit_chb' or self.select == 'chb_fpred'):

            self.model_type = 'class'
            pre = {'loss':'squared_error',
                'lr':0.1,
                'epoch':100,
                'max_depth':None,
                'min_samples_leaf':20,
                'max_bins':255,
                'validation_fraction':0.1,
                'n_iter_no_change':10,
            }

            self.model = HistGradientBoostingClassifier(loss=self.sfp(args,pre,'loss'),
                                                learning_rate=self.sfp(args,pre,'lr'),
                                                max_iter=self.sfp(args,pre,'epoch'),
                                                max_depth=self.sfp(args,pre,'max_depth'),
                                                min_samples_leaf=self.sfp(args,pre,'min_samples_leaf'),
                                                max_bins=self.sfp(args,pre,'max_bins'),
                                                validation_fraction=self.sfp(args,pre,'validation_fraction'),
                                                n_iter_no_change=self.sfp(args,pre,'n_iter_no_change')

                                    )

          
'''

Module Corpus 

'''

dict_slensemble = {
    
                'fit_rada':['create a regression adaboost model',
                             'create AdaBoost regression model',
                             'create adaboost regressor',
                             'fit adaboost regression model',
                             'fit adaboost regressor',
                             'train adaboost regressor',
                             'train Adaboost regressor',
                             'train Adaboost regression model'],
                    
                'rada_fpred' : ['fit and predict adaboost regression model',
                             'create and predict adaboost regression model',
                             'fit and predict adabost regressor',
                             'create and predict adaboost regressor',
                             'fit_predict adaboost regressor',
                             'fit_predict adaboost regression model',
                             'predict AdaBoost regressor',
                             ],

                'fit_cada':['create a classification adaboost model',
                            'create AdaBoost classification model',
                            'create adaboost classifier',
                            'fit adaboost classifier model',
                            'fit adaboost classifier',
                            'train adaboost classifier',
                            'train Adaboost classifier',
                            'train Adaboost classification model',
                            'adaboost classification',
                            'AdaBoost classifier',
                            'adaboost classifier',
                            'fit adaboost classification model'],

                'cada_fpred' : ['fit and predict adaboost classification model',
                             'create and predict adaboost classification model',
                             'fit and predict AdaBoost classification model',
                             'create and predict AdaBoost classifier',
                             'fit and predict adabost classifier',
                             'create and predict adaboost classifier',
                             'fit_predict adaboost classifier',
                             'fit_predict adaboost classification model',
                             'predict adaboost classifier',
                             'predict AdaBoost classifier',
                             ],


                'fit_rrf':['create a regression randomforest model',
                             'create RandomForest regression model',
                             'create randomforest regressor',
                             'fit randomforest regression model',
                             'fit randomforest regressor',
                             'train randomforest regressor',
                             'train randomfores regressor',
                             'train randomforest regression model'],
                    
                'rrf_fpred' : ['fit and predict randomforest regression model',
                             'create and predict randomforest regression model',
                             'fit and predict randomforest regressor',
                             'create and predict randomforest regressor',
                             'fit_predict randomforest regressor',
                             'fit_predict randomforest regression model',
                             'predict RandomForest regressor',
                             ],

                'fit_crf':['create a classification randomforest model',
                            'create RandomForest classification model',
                            'create randomforest classifier',
                            'fit randomforest classifier model',
                            'fit randomforest classifier',
                            'train randomforest classifier',
                            'train randomforest classifier',
                            'train randomforest classification model',
                            'randomforest classification',
                            'RandomForest classifier',
                            'randomforest classifier',
                            'fit randomforest classification model'],

                'crf_fpred' : ['fit and predict randomforest classification model',
                             'create and predict randomforest classification model',
                             'fit and predict RandomForest classification model',
                             'create and predict RandomForest classifier',
                             'fit and predict randomforest classifier',
                             'create and predict randomforest classifier',
                             'fit_predict randomforest classifier',
                             'fit_predict randomforest classification model',
                             'predict randomforest classifier',
                             'predict RandomForest classifier',
                             ],

                'fit_rhb':['create a regression histogram gradient boosting model',
                             'create gradient histogram boosting regression model',
                             'create histgradboost regressor',
                             'fit histgradboost regression model',
                             'fit histgradboost regressor',
                             'train histgradboost regressor',
                             'train HistGradBoost regressor',
                             'train HistGradBoost regression model'],
                    
                'rhb_fpred' : ['fit and predict histgradboost regression model',
                             'create and predict histgradboost regression model',
                             'fit and predict histgradboost regressor',
                             'create and predict histgradboost regressor',
                             'fit_predict histgradboost regressor',
                             'fit_predict histgradboost regression model',
                             'predict HistGradBoost regressor',
                             ],


                'fit_chb':['create a classification histgradboost model',
                            'create HistGradBoost classification model',
                            'create histgradboost classifier',
                            'fit histgradboost classifier model',
                            'fit histgradboost classifier',
                            'train histgradboost classifier',
                            'train histgradboost classifier',
                            'train histgradboost classification model',
                            'histgradboost classification',
                            'HistGradBoost classifier',
                            'histgradboost classifier',
                            'fit histgradboost classification model'],

                'chb_fpred' : ['fit and predict histgradboost classification model',
                             'create and predict histgradboost classification model',
                             'fit and predict HistGradBoost classification model',
                             'create and predict HistGradBoost classifier',
                             'fit and predict histgradboost classifier',
                             'create and predict histgradboost classifier',
                             'fit_predict histgradboost classifier',
                             'fit_predict histgradboost classification model',
                             'predict histgradboost classifier',
                             'predict histgradboost classifier',
                             ],
        

                }


info_slensemble = {
    
                'fit_rada':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble regression',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"An Adaboost regressor is a variant of the Adaboost algorithm that is used for regression tasks instead of classification tasks. It works by iteratively training weak regressors on different subsets of the training data and then combining them into a single strong regressor. Similar to Adaboost for classification, Adaboost regressor assigns higher weights to mispredicted examples and lower weights to correctly predicted examples. The final model is a weighted combination of all the weak regressors, with each regressor's weight determined by its accuracy.",
                            'token_compat':'data features target',
                            'arg_compat':'estimator n_estimators lr loss'},

                'rada_fpred':{'module':'slensemble',
                            'action':'model predict',
                            'topic':'ensemble regression',
                            'subtopic':'model prediction',
                            'input_format':'pd.DataFrame',
                            'description':"An Adaboost regressor is a variant of the Adaboost algorithm that is used for regression tasks instead of classification tasks. It works by iteratively training weak regressors on different subsets of the training data and then combining them into a single strong regressor. Similar to Adaboost for classification, Adaboost regressor assigns higher weights to mispredicted examples and lower weights to correctly predicted examples. The final model is a weighted combination of all the weak regressors, with each regressor's weight determined by its accuracy.",
                            'token_compat':'data features target',
                            'arg_compat':'estimator n_estimators lr loss'},

                'fit_cada':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble classification',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"Adaboost (short for Adaptive Boosting) is a machine learning algorithm that combines multiple weak classifiers to create a strong classifier. It is a boosting algorithm that works by iteratively training weak classifiers on different subsets of the training data and then combining them into a single strong classifier. Adaboost assigns higher weights to misclassified examples and lower weights to correctly classified examples, thereby emphasizing the importance of the misclassified examples in the subsequent rounds of training. The final model is a weighted combination of all the weak classifiers, with each classifier's weight determined by its accuracy.",
                            'token_compat':'data features target',
                            'arg_compat':'estimator n_estimators lr'},

                'cada_fpred':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble classification',
                            'subtopic':'model predict',
                            'input_format':'pd.DataFrame',
                            'description':"Adaboost (short for Adaptive Boosting) is a machine learning algorithm that combines multiple weak classifiers to create a strong classifier. It is a boosting algorithm that works by iteratively training weak classifiers on different subsets of the training data and then combining them into a single strong classifier. Adaboost assigns higher weights to misclassified examples and lower weights to correctly classified examples, thereby emphasizing the importance of the misclassified examples in the subsequent rounds of training. The final model is a weighted combination of all the weak classifiers, with each classifier's weight determined by its accuracy.",
                            'token_compat':'data features target',
                            'arg_compat':'estimator n_estimators lr'},



                'fit_rrf':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble regression',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"Random forest regressor is a machine learning algorithm that is used for regression tasks. It works by constructing multiple decision trees on different subsets of the training data and then combining them into a single model. Each decision tree in the random forest is trained on a random subset of the features and a random subset of the training data, which helps to reduce overfitting and improve generalization. The final prediction is made by averaging the predictions of all the decision trees in the forest.",
                            'token_compat':'data features target',
                            'arg_compat':'n_estimators criterion min_samples_leaf min_samples_split max_depth bootstrap oob_score'},

                'rrf_fpred':{'module':'slensemble',
                            'action':'model predict',
                            'topic':'ensemble regression',
                            'subtopic':'model prediction',
                            'input_format':'pd.DataFrame',
                            'description':"Random forest regressor is a machine learning algorithm that is used for regression tasks. It works by constructing multiple decision trees on different subsets of the training data and then combining them into a single model. Each decision tree in the random forest is trained on a random subset of the features and a random subset of the training data, which helps to reduce overfitting and improve generalization. The final prediction is made by averaging the predictions of all the decision trees in the forest.",
                            'token_compat':'data features target',
                            'arg_compat':'n_estimators criterion min_samples_leaf min_samples_split max_depth bootstrap oob_score'},

                'fit_crf':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble classification',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"Random forest classifier is a machine learning algorithm that is used for classification tasks. It works by constructing multiple decision trees on different subsets of the training data and then combining them into a single model. Each decision tree in the random forest is trained on a random subset of the features and a random subset of the training data, which helps to reduce overfitting and improve generalization. The final prediction is made by taking the majority vote of all the decision trees in the forest.",
                            'token_compat':'data features target',
                            'arg_compat':'n_estimators criterion min_samples_leaf min_samples_split max_depth bootstrap oob_score'},

                'crf_fpred':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble classification',
                            'subtopic':'model predict',
                            'input_format':'pd.DataFrame',
                            'description':"Random forest classifier is a machine learning algorithm that is used for classification tasks. It works by constructing multiple decision trees on different subsets of the training data and then combining them into a single model. Each decision tree in the random forest is trained on a random subset of the features and a random subset of the training data, which helps to reduce overfitting and improve generalization. The final prediction is made by taking the majority vote of all the decision trees in the forest.",
                            'token_compat':'data features target',
                            'arg_compat':'n_estimators criterion min_samples_leaf min_samples_split max_depth bootstrap oob_score'},


                'fit_rhb':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble regression',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"HistGradientBoostingRegressor is a machine learning algorithm that is used for regression tasks. It is a gradient boosting algorithm that uses histograms to improve the speed and efficiency of the training process. It works by constructing an ensemble of decision trees, where each tree is trained to predict the residual error of the previous tree. The algorithm uses gradient descent to minimize the loss function, which is typically the mean squared error. HistGradientBoostingRegressor is particularly effective for large datasets with many features, as it can handle high-dimensional data more efficiently than other gradient boosting algorithms.",
                            'token_compat':'data features target',
                            'arg_compat':'loss lr max_iter max_depth min_samples_leaf max_bins validation_fraction n_iter_no_change'},

                'rhb_fpred':{'module':'slensemble',
                            'action':'model predict',
                            'topic':'ensemble regression',
                            'subtopic':'model prediction',
                            'input_format':'pd.DataFrame',
                            'description':"HistGradientBoostingRegressor is a machine learning algorithm that is used for regression tasks. It is a gradient boosting algorithm that uses histograms to improve the speed and efficiency of the training process. It works by constructing an ensemble of decision trees, where each tree is trained to predict the residual error of the previous tree. The algorithm uses gradient descent to minimize the loss function, which is typically the mean squared error. HistGradientBoostingRegressor is particularly effective for large datasets with many features, as it can handle high-dimensional data more efficiently than other gradient boosting algorithms.",
                            'token_compat':'data features target',
                            'arg_compat':'loss lr max_iter max_depth min_samples_leaf max_bins validation_fraction n_iter_no_change'},

                'fit_chb':{'module':'slensemble',
                            'action':'train model',
                            'topic':'ensemble classification',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"HistGradientBoostingClassifier is a machine learning algorithm that is used for classification tasks. It is similar to HistGradientBoostingRegressor, but it is designed for predicting categorical outcomes rather than numerical values. Like its regression counterpart, HistGradientBoostingClassifier uses histograms to improve the speed and efficiency of the training process. It works by constructing an ensemble of decision trees, where each tree is trained to predict the residual error of the previous tree. The algorithm uses gradient descent to minimize the loss function, which is typically the log loss or cross-entropy loss. HistGradientBoostingClassifier is particularly effective for large datasets with many features, as it can handle high-dimensional data more efficiently than other gradient boosting algorithms.",
                            'token_compat':'data features target',
                            'arg_compat':'loss lr max_iter max_depth min_samples_leaf max_bins validation_fraction n_iter_no_change'},

                'chb_fpred':{'module':'slensemble',
                            'action':'model predict',
                            'topic':'ensemble classification',
                            'subtopic':'model prediction',
                            'input_format':'pd.DataFrame',
                            'description':"HistGradientBoostingClassifier is a machine learning algorithm that is used for classification tasks. It is similar to HistGradientBoostingRegressor, but it is designed for predicting categorical outcomes rather than numerical values. Like its regression counterpart, HistGradientBoostingClassifier uses histograms to improve the speed and efficiency of the training process. It works by constructing an ensemble of decision trees, where each tree is trained to predict the residual error of the previous tree. The algorithm uses gradient descent to minimize the loss function, which is typically the log loss or cross-entropy loss. HistGradientBoostingClassifier is particularly effective for large datasets with many features, as it can handle high-dimensional data more efficiently than other gradient boosting algorithms.",
                            'token_compat':'data features target',
                            'arg_compat':'loss lr max_iter max_depth min_samples_leaf max_bins validation_fraction n_iter_no_change'},

                }

        

configure_slensemble = {'corpus':dict_slensemble,'info':info_slensemble}               