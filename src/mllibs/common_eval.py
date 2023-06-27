from sklearn.base import clone

''' 

COMMON FUNCTIONS / CLASSES

'''

class eval_methods:
    
    # function which fits on entire dataset
    
    @staticmethod
    def fit_funct_all(features,target,model):
        model.fit(features,target)
        return features,target,model
    
    # function which fits on entire dataset
    
    @staticmethod
    def fitpred_funct_all(features,target,model):
        model.fit(features,target)
        y_pred = model.predict(features)
        return features,target,model,y_pred
    
    # functions which fits on training & predicts on training & test sets

    @staticmethod
    def fitpred_funct_tts(features,target,model,groups):

        # training data
        train_X = features.iloc[groups['train'].index,:]
        train_y = target.iloc[groups['train'].index]
        model.fit(train_X,train_y)

        # predict on train
        y_pred_tr = model.predict(train_X)
        
        # predict on test
        test_X = features.iloc[groups['test'].index,:]
        test_y = target.iloc[groups['test'].index]
        y_pred_te = model.predict(test_X)

        return train_X,train_y,test_X,test_y,model,y_pred_tr,y_pred_te
    
    # functions which fits on training data only

    @staticmethod
    def fit_funct_tts(features,target,model,groups):

        # training data
        train_X = features.iloc[groups['train'].index,:]
        train_y = target.iloc[groups['train'].index]
        model.fit(train_X,train_y)

        return train_X,train_y,model
    

    # function which [fits] on training set in kfold cv
    
    @staticmethod
    def fit_funct_kfold(features,target,model,cv):

        cv_preds = []; cv_y = []; train_index = []; models = []

        # cross validation loop
        for train_idx,_ in cv.split(features):

            X_train = features.iloc[train_idx,:]
            y_train = target.iloc[train_idx]
            lmodel = clone(model)
            lmodel.fit(X_train,y_train)
            y_pred = lmodel.predict(X_train)

            cv_preds.append(y_pred)       # prediction on training data folds
            cv_y.append(y_train)          # save fold target values
            train_index.append(train_idx)  # save training indicies of folds
            models.append(lmodel)        # save all models in folds

        return train_index,cv_y,cv_preds,models

    # function which [fits] & [predicts] on train/test set in kfold cv

    @staticmethod
    def fitpred_funct_kfold(
        features,target,model,cv):
        
        cv_ytr = []; cv_yte = []; 
        cv_preds_tr = []; cv_preds_te = []
        test_index = []; train_index = []; 
        models = []

        # cross validation loop
        for train_idx, test_idx in cv.split(features):

            X_train = features.iloc[train_idx,:]
            y_train = target.iloc[train_idx]
            X_test = features.iloc[test_idx,:]
            y_test = target.iloc[test_idx]

            lmodel = clone(model)
            lmodel.fit(X_train,y_train)
            y_pred_tr = lmodel.predict(X_train)
            y_pred_te = lmodel.predict(X_test)

            cv_ytr.append(y_train)         # save train fold target values
            cv_yte.append(y_test)          # save test fold target values
            cv_preds_tr.append(y_pred_tr)  # save train fold prediction 
            cv_preds_te.append(y_pred_te)  # save test fold prediction

            train_index.append(train_idx) # save training indicies of folds
            test_index.append(test_idx)   # save test indicies of fo lds

            models.append(lmodel)       # save models trained on training folds

        return train_index,test_index,cv_ytr,cv_yte,cv_preds_tr,cv_preds_te,models
    


from sklearn.metrics import classification_report 
from sklearn.metrics import mean_squared_error
from mllibs.nlpi import nlpi


'''

Base Evaluation Class

'''

class eval_base(nlpi):
    
    def __init__(self,nlp_config,name):
        self.name = name
        self.nlp_config = nlp_config 
        self.select = None
        self.args = None

        self.model_type = None
        self.model = None

    '''
    
    Static Methods
    
    '''
        
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

    def set_model(self,args):
        raise NotImplementedError()


    def sel(self,args:dict):

        ''' 
        
        COMMON SETTINGS FOR MULTIPLE ACTIVATION FUNCTIONS
        
        '''
    
        # define instance variables
        self.select = args['pred_task']      # activation function name
        self.data_name = args['data_name']   # key of used data
        self.args = args                     # all argument data
        self.splits = args['splits']         # data splitting key (cv or tts)
        
        # check feature and target variable have been defined

        set_feat = False; set_target = False
        if(len(self.args['features']) > 1):
            features = self.args['data'][self.args['features']]  # dataframe
            set_feat = True
        if(len(self.args['target']) == 1):
            target = self.args['data'][self.args['target'][0]] # series
            set_target = True
        
        if(set_feat is not True and set_target is not True):
            print('features and target not set correctly')
            return 

                
        ''' 
        
        FIT MODEL ON DATA 
        
        '''

        # check if split name is present in data 
        
        afname = args['data_name'][0]   # name of data in nlpi.data
        afname_splits = nlpi.data[afname]['splits']   # dictionary of splits

        # no split has been made yet

        split_id = None
        if(len(afname_splits) == 0):
            split_id = None

        # split has been made
        else:
        
            made_splits = list(afname_splits.keys())[0]

            if(self.splits in made_splits):
                if('tts' in self.splits):
                    split_id = 'tts'
                elif('kfold' in self.splits):
                    split_id = 'kfold'
                elif('skfold' in self.splits):
                    split_id = 'skfold'

        # set model & model type
        self.set_model(args)

        
        ''' 
        
        3. Main Selection Functions 
        
        '''

        # [1] Train test split train on [X_train] predict on [X_test]

        if(split_id == 'tts' and '_fpred' in self.select):
            
            # get training / test groups
            groups = dict(tuple(self.data[afname]['splits'][self.splits].to_frame().groupby(by='tts')))

            X_train,y_train,X_test,y_test,model,yp_train,yp_test = eval_methods.fitpred_funct_tts(features,target,self.model,groups)

            # evaluation metric (RMSE)
            if(self.model_type == 'reg'):
                criterion_tr = mean_squared_error(y_train,yp_train,squared=False)
                criterion_te = mean_squared_error(y_test,yp_test,squared=False)

            # evaluation metric (classification report)
            elif(self.model_type == 'class'):
                criterion_tr = classification_report(y_train,yp_train)
                criterion_te = classification_report(y_test,yp_test)

            # storage
            nlpi.memory_output.append({
                                       'X_train':X_train,
                                       'y_train':y_train,
                                       'X_test':X_test,
                                       'y_test':y_test,
                                       'model':model,
                                       'criterion_tr':criterion_tr,
                                       'criterion_te':criterion_te,
                                       'yp_train':yp_train,
                                       'yp_test':yp_test
                                        }) 

        # [2] train test split train on [train] 

        elif(split_id == 'tts' and '_fpred' not in self.select):

            # get training / test groups
            groups = dict(tuple(self.data[afname]['splits'][self.splits].to_frame().groupby(by='tts')))

            X_train,y_train,X_test,y_test,model = eval_methods.fit_funct_tts(features,target,self.model,groups)

            # storage
            nlpi.memory_output.append({
                                       'X_train':X_train,
                                       'y_train':y_train,
                                       'model':model,
                                        }) 

        # [3] train model on all data & predict model on all data

        elif(split_id == None and '_fpred' in self.select):
            
            features,target,model,y_pred = eval_methods.fitpred_funct_all(features,target,self.model)

            # select evaluation criterion based on model type
            if(self.model_type == 'reg'):
                criterion = mean_squared_error(target,y_pred,squared=False)
            elif(self.model_type == 'class'):
                criterion = classification_report(target,y_pred)

            # storage
            nlpi.memory_output.append({
                                       'features':features,
                                       'target':target,
                                       'model':model,
                                       'criterion':criterion,
                                       'yp_test':y_pred
                                        }) 
            
        # [4] train model on all data

        elif(split_id == None and '_fpred' not in self.select):

            features,target,model = eval_methods.fit_funct_all(features,target,self.model)

            # storage
            nlpi.memory_output.append({
                                       'features':features,
                                       'target':target,
                                       'model':model,
                                        })  

        # [5] scikit-learn fold methods

        elif(split_id == 'kfold' or split_id == 'skfold'):

            # train model on [cv-train fold] & evaluate on remaining [cv-test fold]
            # repeated for number of folds

            # recall cv splitter
            cv = self.data[afname]['splits'][self.splits]

            if('_fpred' in self.select):       

                # recall cv splitter
                cv = self.data[afname]['splits'][self.splits]
            
                X_train,X_test,y_train,y_test,yp_train,yp_test,models = eval_methods.fitpred_funct_kfold(features,target,self.model,cv)

                # criterion evaluation for each [training] fold

                criterion_tr = []
                if(self.model_type == 'reg'):

                    for ii,fold in enumerate(yp_train):
                        y_fold = y_train[ii]
                        yp_fold = yp_train[ii]
                        criterion_tr.append(mean_squared_error(y_fold,yp_fold,squared=False))

                elif(self.model_type == 'class'):

                    for ii,fold in enumerate(yp_train):
                        y_fold = y_train[ii]
                        yp_fold = yp_train[ii]
                        criterion_tr.append(classification_report(y_fold,yp_fold))

                # criterion evaluation for each [test] fold

                criterion_te = []
                if(self.model_type == 'reg'):

                    for ii,fold in enumerate(yp_test):
                        y_fold = y_test[ii]
                        yp_fold = yp_test[ii]
                        criterion_te.append(mean_squared_error(y_fold,yp_fold,squared=False))

                elif(self.model_type == 'class'):

                    for ii,fold in enumerate(yp_test):
                        y_fold = y_test[ii]
                        yp_fold = yp_test[ii]
                        criterion_te.append(classification_report(y_fold,yp_fold))


                # storage
                nlpi.memory_output.append({'X_train':X_train,  # index of training indicies for each fold
                                           'X_test':X_test,    # index of training indicies for each fold
                                           'y_train':y_train,  # index of test indicies for each fold
                                           'y_test':y_test,    # index of test indicies for each fold
                                           'features':features, # input features
                                           'target':target,    # input target variaboe
                                           'model':models,     # input models
                                           'criterion_tr':criterion_tr,
                                           'criterion_te':criterion_te
                                            })  

            # train model on [cv-train fold] only

            elif('_fpred' not in self.select): 

                X_train,y_train,yp_train,models = eval_methods.fit_funct_kfold(features,target,self.model,cv)

                # criterion evaluation for each fold

                criterion = []
                if(self.model_type == 'reg'):

                    for ii,fold in enumerate(yp_train):
                        y_fold = y_train[ii]
                        yp_fold = yp_train[ii]
                        criterion.append(mean_squared_error(y_fold,yp_fold,squared=False))

                elif(self.model_type == 'class'):

                    for ii,fold in enumerate(yp_train):
                        y_fold = y_train[ii]
                        yp_fold = yp_train[ii]
                        criterion.append(classification_report(y_fold,yp_fold))

                # storage
                nlpi.memory_output.append({'X_train':X_train,  # index of training indicies for each fold
                                           'y_train':y_train,  # index of training indicies for each fold
                                           'features':features, # feature matrix used in input
                                           'target':target,     # target vector used in input
                                           'model':models,      # kfold models
                                           'yp_train':yp_train,  # model prediction for each training fold
                                           'criterion_tr':criterion # evaluation metric for each [training] fold
                                            })  