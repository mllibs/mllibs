from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from mllibs.common_eval import eval_base


'''

DECISION TREE MODELS

'''

# sklearn decision tree models from [.tree]

# requires parent evaluation class 
# we only need to replace class name & set model attribute/method

class sltree(eval_base):

    def __init__(self,nlp_config):
        self.name = 'sltree'
        eval_base.__init__(self,nlp_config,self.name)


    def set_model(self,args):

        # select model (7 variations - 5 regressors / 2 classifiers)

        if(self.select == 'fit_rdc' or self.select == 'rdc_fpred'):   

            self.model_type = 'reg'
            pre = {"criterion":'squared_error',
                   'splitter':'best',
                   'max_depth':None,
                   'min_samples_leaf':1,
                   'min_samples_split':2,
                   'max_features':None,
                   'rs':None
                   }
            
            self.model = DecisionTreeRegressor(criterion=self.sfp(args,pre,'criterion'),
                                               splitter=self.sfp(args,pre,'splitter'),
                                               max_depth=self.sfp(args,pre,'max_depth'),
                                               min_samples_leaf=self.sfp(args,pre,'min_samples_leaf'),
                                               min_samples_split=self.sfp(args,pre,'min_samples_split'),
                                               max_features=self.sfp(args,pre,'max_features'),
                                               random_state=self.sfp(args,pre,'rs')
                                              )
                                               
        elif(self.select == 'fit_cdc' or self.select == 'cdc_fpred'):

            self.model_type = 'class'
            pre = {"criterion":'gini',
                   'splitter':'best',
                   'max_depth':None,
                   'min_samples_leaf':1,
                   'min_samples_split':2,
                   'max_features':None,
                   'rs':None
                   }
            
            self.model = DecisionTreeClassifier(criterion=self.sfp(args,pre,'criterion'),
                                               splitter=self.sfp(args,pre,'splitter'),
                                               max_depth=self.sfp(args,pre,'max_depth'),
                                               min_samples_leaf=self.sfp(args,pre,'min_samples_leaf'),
                                               min_samples_split=self.sfp(args,pre,'min_samples_split'),
                                               max_features=self.sfp(args,pre,'max_features'),
                                               random_state=self.sfp(args,pre,'rs')
                                              )
            
'''

Module Corpus 

'''

dict_sltree = { 'fit_rdc' : ['create a regression decision tree model',
                             'create a decision tree regressor',
                             'create decision tree regressor',
                             'create DecisionTree Regressor',
                             'create DecisionTreeRegressor',
                             'create a decision tree regression model',
                             'fit decision tree regression model',
                             'fit regression decision tree model',
                             'fit regressor decision tree model',
                             'fit DecisionTreeRegressor',
                             'regressor decision tree model',
                             'decision tree regression model',
                             'train regression decision tree model',
                             'train decision tree regressor',
                             'train decision tree regressor model',
                             'train DecisionTreeRegressor'],
               
                'rdc_fpred': ['fit and predict DecisionTreeRegressor',
                              'fit and predict decision tree regressor',
                              'fit and predict decision tree regression model',
                              'create and predict decision tree regressor',
                              'craete and predict DecisionTreeRegressor',
                              'fit_predict DecisionTreeRegressor',
                              'fit_predict decision tree regressor',
                              'fit_predict DecisionTree regressor',
                              'predict DecisionTree regressor',
                              ],

                'fit_cdc' : ['create a classification decision tree model',
                             'create a decision tree classifier',
                             'create decision tree classifier',
                             'create DecisionTree classifier',
                             'create DecisionTreeClassifier',
                             'create a decision tree classification model',
                             'fit decision tree classification model',
                             'fit classification decision tree model',
                             'fit classification decision tree model',
                             'fit DecisionTreeClassifier',
                             'classifier decision tree model',
                             'decision tree classifier model',
                             'train classifier decision tree model',
                             'train decision tree classifier',
                             'train decision tree classifier model',
                             'train DecisionTreeClassifier'],
               
                'cdc_fpred': ['fit and predict DecisionTreeClassifier',
                              'fit and predict decision tree classifier',
                              'fit and predict decision tree classification model',
                              'create and predict decision tree classifier',
                              'craete and predict DecisionTreeClassifier',
                              'fit_predict DecisionTreeClassifier',
                              'fit_predict decision tree classifier',
                              'fit_predict DecisionTree classifier',
                              'predict DecisionTree classifier',
                              ]
}

info_sltree = {
    
                  'fit_rdc':{'module':'sltree',
                            'action':'train model',
                            'topic':'decision tree regression',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"A decision tree regression model is a type of machine learning model that uses a decision tree algorithm to make predictions on continuous numerical values. It works by recursively partitioning the input space into smaller regions based on the values of the input features, and then assigning a constant value (usually the mean or median) to each region as the predicted output. The decision tree algorithm determines the optimal splits and thresholds for partitioning the data based on certain criteria, such as minimizing the variance of the predicted values within each region. This allows the model to capture non-linear relationships between the input features and the target variable, making it suitable for regression tasks.",
                            'token_compat':'data features target',
                            'arg_compat':'criterion splitter max_depth min_samples_leaf min_samples_split max_features rs'},


                  'rdc_fpred':{'module':'sltree',
                              'action':'model predict',
                              'topic':'decision tree regression',
                              'subtopic':'model prediction',
                              'input_format':'pd.DataFrame',
                              'description':"A decision tree regression model is a type of machine learning model that uses a decision tree algorithm to make predictions on continuous numerical values. It works by recursively partitioning the input space into smaller regions based on the values of the input features, and then assigning a constant value (usually the mean or median) to each region as the predicted output. The decision tree algorithm determines the optimal splits and thresholds for partitioning the data based on certain criteria, such as minimizing the variance of the predicted values within each region. This allows the model to capture non-linear relationships between the input features and the target variable, making it suitable for regression tasks.",
                            'token_compat':'data features target',
                            'arg_compat':'criterion splitter max_depth min_samples_leaf min_samples_split max_features rs'},


                  'fit_cdc':{'module':'sltree',
                            'action':'train model',
                            'topic':'decision tree classification',
                            'subtopic':'model training',
                            'input_format':'pd.DataFrame',
                            'description':"A decision tree classification model is a type of machine learning model that uses a decision tree algorithm to make predictions on categorical or discrete values. Instead of predicting continuous numerical values, the decision tree classification model predicts the class or category that an input data point belongs to. It works by recursively partitioning the input space into smaller regions based on the values of the input features, and then assigning a class label to each region as the predicted output. The decision tree algorithm determines the optimal splits and thresholds for partitioning the data based on certain criteria, such as maximizing the purity or homogeneity of the predicted classes within each region. This allows the model to capture complex decision boundaries and classify new data points based on their feature values. ",
                            'arg_compat':'criterion splitter max_depth min_samples_leaf min_samples_split max_features rs'},

                  'cdc_fpred':{'module':'sltree',
                              'action':'model predict',
                              'topic':'decision tree classification',
                              'subtopic':'model prediction',
                              'input_format':'pd.DataFrame',
                              'description':"A decision tree classification model is a type of machine learning model that uses a decision tree algorithm to make predictions on categorical or discrete values. Instead of predicting continuous numerical values, the decision tree classification model predicts the class or category that an input data point belongs to. It works by recursively partitioning the input space into smaller regions based on the values of the input features, and then assigning a class label to each region as the predicted output. The decision tree algorithm determines the optimal splits and thresholds for partitioning the data based on certain criteria, such as maximizing the purity or homogeneity of the predicted classes within each region. This allows the model to capture complex decision boundaries and classify new data points based on their feature values.",
                            'token_compat':'data features target',
                            'arg_compat':'criterion splitter max_depth min_samples_leaf min_samples_split max_features rs'},

}

configure_sltree = {'corpus':dict_sltree,'info':info_sltree}     
