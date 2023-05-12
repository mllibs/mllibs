from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel,sigmoid_kernel
from sklearn.base import clone
from collections import OrderedDict
import numpy as np
import pandas as pd
import zipfile

import nltk
nltk.download('wordnet')

wordn = '/usr/share/nltk_data/corpora/wordnet.zip'
wordnt = '/usr/share/nltk_data/corpora/'

with zipfile.ZipFile(wordn,"r") as zip_ref:
    zip_ref.extractall(wordnt)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


'''


NLPM class

- combine together all extension modules
- create machine learning models for task prediction



'''



class nlpm:
    
    def __init__(self):
        self.task_dict = {} # stores the input task variation dictionary (prepare)
        self.functions = {} # stores model associate function class (prepare) 
        
    # Convert lists of text to dataframe with label
    
    @staticmethod
    def nltk_tokeniser(text):
        return word_tokenize(text)
    
    @staticmethod
    def lists_to_frame(lsts):

        dict_txt = {'text':[],'class':[]}
        for i,lst in enumerate(lsts):
            dict_txt['text'] += lst
            for j in lst:
                dict_txt['class'].append(i)

        return pd.DataFrame(dict_txt)

    ''' 
    
    # load module & prepare module content data
    
    '''     
            
    def load(self,modules:list):
            
        if(type(modules) is list):
            
            print('loading modules ...')
            
            # combined module information/option dictionaries
            
            lst_module_info = []; lst_corpus = []; dict_task_names = {}
            for module in modules:  
                
                self.functions[module.name] = module          # store module functions
                
                # combine corpuses of modules
                tdf_corpus = module.nlp_config['corpus']
                df_corpus = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_corpus.items()]))          
                dict_task_names[module.name] = list(df_corpus.columns)  # save order of module task names
                
                lst_corpus.append(df_corpus)
                self.task_dict[module.name] = tdf_corpus     # save corpus
                
                # combine info of different modules
                opt = module.nlp_config['info']     # already defined task corpus
                tdf_opt = pd.DataFrame(opt)
                df_opt = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_opt.items()]))
                lst_module_info.append(df_opt)

            ''' Step 1 : Create Task Corpuses (dataframe) '''
                
            # task corpus (contains no label)
            corpus = pd.concat(lst_corpus,axis=1) 
            
            ''' Step 2 : Create Task information dataframe '''
            # create combined_opt : task information data
            
            # task information options
            combined_opt = pd.concat(lst_module_info,axis=1)
            combined_opt = combined_opt.T.sort_values(by='module')
            combined_opt_index = combined_opt.index
            
            
            ''' Step 3 : Create Module Corpus Labels '''         
            print('making module summary labels...')

            # note groupby (alphabetically module order) (module order setter)
            module_groupby = dict(tuple(combined_opt.groupby(by='module')))
            unique_module_groupby = list(module_groupby.keys())  # [eda,loader,...]

            for i in module_groupby.keys():
                ldata = module_groupby[i]
                ldata['task_id'] = range(0,ldata.shape[0])

            df_opt = pd.concat(module_groupby).reset_index(drop=True)
            df_opt.index = combined_opt_index
            
            # task orders
            self.mod_order = unique_module_groupby
            self.label = {}
            
            # generate task labels    
            encoder = LabelEncoder()
            df_opt['gtask_id'] = range(df_opt.shape[0])
            self.label['gt'] = list(combined_opt_index)
            
            encoder = clone(encoder)
            df_opt['module_id'] = encoder.fit_transform(df_opt['module'])   
            self.label['ms'] = encoder.classes_
            
            encoder = clone(encoder)
            df_opt['action_id'] = encoder.fit_transform(df_opt['action'])
            self.label['act'] = encoder.classes_
            
            encoder = clone(encoder)
            df_opt['topic_id'] = encoder.fit_transform(df_opt['topic'])
            self.label['top'] = encoder.classes_
            
            encoder = clone(encoder)
            df_opt['subtopic_id'] = encoder.fit_transform(df_opt['subtopic'])
            self.label['sub'] = encoder.classes_
            
            # Main Data 
            self.mod_summary = df_opt
            
            # created self.mod_summary
            # created self.label
            
            
            ''' Make Module Task Corpus '''
            
            lst_modules = dict(list(df_opt.groupby('module_id')))
            module_task_corpuses = OrderedDict()   # store module corpus
            module_task_names = {}                 # store module task names
            
            for ii,i in enumerate(lst_modules.keys()):
                
                columns = list(lst_modules[i].index)      # module task names
                column_vals =  corpus[columns].dropna()
                module_task_names[unique_module_groupby[i]] = columns

                lst_module_classes = []
                for ii,task in enumerate(columns):
                    ldf_task = column_vals[task].to_frame()
                    ldf_task['class'] = ii

                    lst_module_classes.append(pd.DataFrame(ldf_task.values))

                tdf = pd.concat(lst_module_classes)
                tdf.columns = ['text','class']
                tdf = tdf.reset_index(drop=True)                
                
                module_task_corpuses[unique_module_groupby[i]] = tdf

            # module task corpus
            self.module_task_name = module_task_names
            self.corpus_mt = module_task_corpuses # dictionaries of dataframe corpuses
                
                
            ''' Make Global Task Selection Corpus '''
        
            def prepare_corpus(group):
            
                lst_modules = dict(list(df_opt.groupby(group)))

                lst_melted = []                
                for ii,i in enumerate(lst_modules.keys()):    
                    columns = list(lst_modules[i].index)
                    column_vals = corpus[columns].dropna()
                    melted = column_vals.melt()
                    melted['class'] = ii
                    lst_melted.append(melted)

                df_melted = pd.concat(lst_melted)
                df_melted.columns = ['task','text','class']
                df_melted = df_melted.reset_index(drop=True)
                
                return df_melted

            # generate task corpuses
            self.corpus_ms = prepare_corpus('module_id') # modue selection dataframe
            self.corpus_gt = prepare_corpus('gtask_id')  # global task dataframe
            self.corpus_act = prepare_corpus('action_id') # action task dataframe
            self.corpus_top = prepare_corpus('topic_id') # topic task dataframe
            self.corpus_sub = prepare_corpus('subtopic_id') # subtopic tasks dataframe
            
            print('done ...')
    
        else:
            raise TypeError('please make input a list of modules!')
            

    
    def mlloop(self,corpus,module_name):
        
        ''' Preprocess text data '''
        
        vect = CountVectorizer()
#         lvect = clone(vect)
        
        # lemmatiser
        lemma = WordNetLemmatizer() 
        
        # define a function for preprocessing
        def clean(text):
            tokens = word_tokenize(text) #tokenize the text
            clean_list = [] 
            for token in tokens:
                clean_list.append(lemma.lemmatize(token)) #lemmatizing and appends to clean_list
            return " ".join(clean_list)# joins the tokens

        # clean corpus
        corpus['text'] = corpus['text'].apply(clean)
        
        ''' Convert text to numeric representation '''
        
        vect.fit(corpus['text']) # input into vectoriser is a series
        vectors = vect.transform(corpus['text']) # sparse matrix
        self.vectoriser[module_name] = vect  # store vectoriser 

        ''' Make training data '''
        X = np.asarray(vectors.todense())
        y = corpus['class'].values.astype('int')

        ''' Train model on numeric corpus '''
        model_lr = LogisticRegression()
        model = clone(model_lr)
        model.fit(X,y)
        self.model[module_name] = model # store model

        # show the training score 
        y_pred = model.predict(X)
        score = model.score(X,y)
        print(module_name,model,'accuracy',round(score,3))
    
    '''
    
    Train Relevant Models
    
    '''
    
    def train(self,type='mlloop'):
                    
        if(type is 'mlloop'):
        
            self.vectoriser = {} # stores vectoriser
            self.model = {}   # storage for models
    
            ''' Create module task model for each module '''

            for ii,(key,corpus) in enumerate(self.corpus_mt.items()):  
                module_name = self.mod_order[ii]
                self.mlloop(corpus,module_name)

            ''' Create Module Selection Model'''
            self.mlloop(self.corpus_ms,'ms')

            ''' Other Models '''

    #         lvect = clone(vect)
    #         self.train_loop(self.corpus_gt,'gt',lvect)
    #         lvect = clone(vect)
    #         self.train_loop(self.corpus_act,'act',lvect)
    #         lvect = clone(vect)
    #         self.train_loop(self.corpus_top,'top',lvect)
    #         lvect = clone(vect)
    #         self.train_loop(self.corpus_sub,'sub',lvect)

            print('models trained...')
        
        
    '''
    
    Model Predictions 
    
    '''
    
    # Inference on sentence to test model
              
    def test(self,name:str,command:str):
        test_phrase = [command]
        Xt_encode = self.vectoriser[name].transform(test_phrase)
        y_pred = self.model[name].predict_proba(Xt_encode)
        return y_pred
    
    def test_name(self,name:str,command:str):
        pred_per = self.test(name,command)     # percentage prediction for all classes
        idx_pred = np.argmax(pred_per)          # index of highest prob 
        pred_name = list(self.model.keys())[idx_pred] # name of highest prob
        return pred_name
    
    # for testing only

    def dtest(self,corpus:str,command:str):
        
        print('available models')
        print(self.model.keys())
        
        prediction = self.test(corpus,command)[0]
        if(corpus in self.label):
            label = list(self.label[corpus])
        else:
            label = list(self.corpus_mt[corpus])
            
        df_pred = pd.DataFrame({'label':label,'prediction':prediction})
        df_pred.sort_values(by='prediction',ascending=False,inplace=True)
        df_pred = df_pred.iloc[:5,:]
        display(df_pred)