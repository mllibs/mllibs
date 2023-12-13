from sklearn.preprocessing import LabelEncoder
from mllibs.tokenisers import nltk_wtokeniser,nltk_tokeniser
from mllibs.nerparser import Parser,ner_model
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel,sigmoid_kernel
from sklearn.base import clone
from collections import OrderedDict
import pickle
import numpy as np
import pandas as pd
import pkgutil
import pkg_resources
import nltk
import io
import csv
import json
# nltk.download('wordnet')

# import zipfile
# wordn = '/usr/share/nltk_data/corpora/wordnet.zip'
# wordnt = '/usr/share/nltk_data/corpora/'

# with zipfile.ZipFile(wordn,"r") as zip_ref:
#      zip_ref.extractall(wordnt)

# parse JSON module data

def parse_json(json_data):

    lst_classes = []; lst_corpus = []; lst_info = []
    for module in json_data['modules']:
        lst_corpus.append(module['corpus'])
        lst_classes.append(module['name'])
        lst_info.append(module['info'])

    return {'corpus':dict(zip(lst_classes,lst_corpus)),
              'info':dict(zip(lst_classes,lst_info))}

'''

NLPM class

> Combine together all extension modules
> Create machine learning models for task prediction

'''

class nlpm:
    
    def __init__(self):
        print('\n[note] initialising nlpm, please load modules using .load(list)')
        self.task_dict = {} # stores the input task variation dictionary (prepare)
        self.modules = {} # stores model associate function class (prepare) 
        self.ner_identifier = {}  # NER tagger (inactive)

    ''' 
    
    load module & prepare module content data
    
    '''     
    
    # group together all module data & construct corpuses
          
    def load(self,modules:list):
            
        print('[note] loading modules ...')
        
        # dictionary for storing model label (text not numeric)
        self.label = {} 
        
        # combined module information/option dictionaries
        
        lst_module_info = []
        lst_corpus = []
        dict_task_names = {}

        for module in modules:  
            
            # get & store module functions
            self.modules[module.name] = module
            
            # get dictionary with corpus
            tdf_corpus = module.nlp_config['corpus']   

            # dictionary of corpus
            df_corpus = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_corpus.items()]))
        
            # module task list
            dict_task_names[module.name] = list(df_corpus.columns)  # save order of module task names

            lst_corpus.append(df_corpus)
            self.task_dict[module.name] = tdf_corpus     # save corpus
            
            # combine info of different modules
            opt = module.nlp_config['info']     # already defined task corpus
            tdf_opt = pd.DataFrame(opt)
            df_opt = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_opt.items()]))
            lst_module_info.append(df_opt)

        # update label dictionary to include loaded modules
        self.label.update(dict_task_names)  
            
        ''' 

        Step 1 : Create Task Corpuses (dataframe) 

        '''
            
        # task corpus (contains no label)
        corpus = pd.concat(lst_corpus,axis=1)
        
        ''' 

        Step 2 : Create Task information dataframe 

        '''
        # create combined_opt : task information data
        
        # task information options
        combined_opt = pd.concat(lst_module_info,axis=1)
        combined_opt = combined_opt.T.sort_values(by='module')
        combined_opt_index = combined_opt.index
        
        
        ''' Step 3 : Create Module Corpus Labels '''         
        print('[note] making module summary labels...')

        # note groupby (alphabetically module order) (module order setter)
        module_groupby = dict(tuple(combined_opt.groupby(by='module')))
        unique_module_groupby = list(module_groupby.keys())  # [eda,loader,...]

        for i in module_groupby.keys():
            ldata = module_groupby[i]
            ldata['task_id'] = range(0,ldata.shape[0])

        df_opt = pd.concat(module_groupby).reset_index(drop=True)
        df_opt.index = combined_opt_index
        
        # module order for ms
        self.mod_order = unique_module_groupby
        
        ''' 

        Step 4 : labels for other models (based on provided info) 

        '''
        
        # generate task labels    
        encoder = LabelEncoder()
        df_opt['gtask_id'] = range(df_opt.shape[0])
        self.label['gt'] = list(combined_opt_index)
        
        encoder = clone(encoder)
        df_opt['module_id'] = encoder.fit_transform(df_opt['module'])   
        self.label['ms'] = list(encoder.classes_)
        
        encoder = clone(encoder)
        df_opt['action_id'] = encoder.fit_transform(df_opt['action'])
        self.label['act'] = list(encoder.classes_)
        
        encoder = clone(encoder)
        df_opt['topic_id'] = encoder.fit_transform(df_opt['topic'])
        self.label['top'] = list(encoder.classes_)
        
        encoder = clone(encoder)
        df_opt['subtopic_id'] = encoder.fit_transform(df_opt['subtopic'])
        self.label['sub'] = list(encoder.classes_)
        
        # Main Summary
        self.mod_summary = df_opt
        
        # created self.mod_summary
        # created self.label
        
        
        ''' 

        Make Module Task Corpus 

        '''
        
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
        # self.module_task_name = module_task_names

        self.label.update(module_task_names) 

        # dictionaries of dataframe corpuses
        self.corpus_mt = module_task_corpuses 
            
            
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
            
    ''' 
    
    MACHINE LEARNING LOOP 
    
    '''
    
    def mlloop(self,corpus:dict,module_name:str):

        # corpus : text [pd.Series] [0-...]
        # class : labels [pd.Series] [0-...]
        
        # lemmatiser
#        lemma = WordNetLemmatizer() 
        
        # define a function for preprocessing
#        def clean(text):
#            tokens = word_tokenize(text) #tokenize the text
#            clean_list = [] 
#            for token in tokens:
#                lemmatizing and appends to clean_list
#                clean_list.append(lemma.lemmatize(token)) 
#            return " ".join(clean_list)# joins the tokens

#         clean corpus
#        corpus['text'] = corpus['text'].apply(clean)
        
        ''' 

        Convert text to numeric representation 

        '''
        
        # vect = CountVectorizer()
#        vect = CountVectorizer(tokenizer=lambda x: word_tokenize(x))
        # vect = CountVectorizer(tokenizer=lambda x: WhitespaceTokenizer().tokenize(x))
        # vect = CountVectorizer(tokenizer=lambda x: nltk_wtokeniser(x),
                               # stop_words=['create'])
        vect = TfidfVectorizer(tokenizer=lambda x: nltk_wtokeniser(x))
        vect.fit(corpus['text']) # input into vectoriser is a series
        vectors = vect.transform(corpus['text']) # sparse matrix
        self.vectoriser[module_name] = vect  # store vectoriser 

        ''' 

        Make training data 

        '''
        
        # X = np.asarray(vectors.todense())
        X = vectors
        y = corpus['class'].values.astype('int')

        ''' 

        Train model on numeric corpus 

        '''
        
        # model_lr = LogisticRegression()
        # model_dt = DecisionTreeClassifier()
        model_rf = RandomForestClassifier()

        # model = clone(model_lr)
        model = clone(model_rf)

        # train model
        model.fit(X,y)
        self.model[module_name] = model # store model
        score = model.score(X,y)
        print(f"[note] training  [{module_name}] [{model}] [accuracy,{round(score,3)}]")
    
    '''
    
    TRAIN RELEVANT MODELS
    
    '''

    # module selection model [ms]
    # > module class models [module name] x n modules
    
    def train(self,type='mlloop'):
                    
        if(type == 'mlloop'):
        
            self.vectoriser = {} # stores vectoriser
            self.model = {}   # storage for models
    
            ''' 

            [1] Create module task model for each module 

            '''

            # for ii,(key,corpus) in enumerate(self.corpus_mt.items()):  
            #     module_name = self.mod_order[ii]
            #     self.mlloop(corpus,module_name)

            ''' 

            [2] Create Module Selection Model

            '''
            # self.mlloop(self.corpus_ms,'ms')

            ''' Other Models '''

            self.mlloop(self.corpus_gt,'gt')
    #         self.mlloop(self.corpus_act,'act')
    #         self.mlloop(self.corpus_top,'top')
    #         self.mlloop(self.corpus_sub,'sub')
    
            # self.toksub_model()
            # self.ner_tokentag_model()  
            self.ner_tagger()

            print('[note] models trained!')
    
    ''' 

    Create multiclass classification model which will determine 
    which approach to utilise for the selection of subset features 

    '''

    def toksub_model(self):

        f = pkgutil.get_data('mllibs', 'corpus/classifier_subset.csv')
        data = pd.read_csv(io.BytesIO(f), encoding='utf8',delimiter=',')

        vectoriser = CountVectorizer(stop_words=['using','use'])
        X = vectoriser.fit_transform(list(data['corpus'])).toarray()
        y = data['label'].values

        model = LogisticRegression().fit(X,y)
        
        self.vectoriser['token_subset'] = vectoriser
        self.model['token_subset'] = model      
        self.label['token_subset'] = ['allbut','only','fromdata','numeric','categorical','all']

    '''
    
    2. NER sentence splitting model 
        
    '''

    # ner tagger for [model parameters],[pp parameters] & [source parameter]

    def ner_tagger(self):
        # f = pkgutil.get_data('mllibs', 'corpus/ner_modelparams_annot.csv')
        path = pkg_resources.resource_filename('mllibs', '/corpus/ner_modelparams_annot.csv')
        df = pd.read_csv(path,delimiter=',')

        parser = Parser()
        model,encoder = ner_model(parser,df)
        self.ner_identifier['model'] = model
        self.ner_identifier['encoder'] = encoder

    # ner tagger for [features] [target] [subset] [data] [other]
        
    def ner_tokentag_model(self):

        # flatten a list of lists
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        typea = ['features','feature list','feature columns','independent']
        typeb = ['target','target column','target variable','dependent']
        typec = ['subset','subset columns']
        typed = ['data','data source','source']
        type_all = typea + typeb + typec + typed

        # tokens = [tokenise(i) for i in type_all]

        tokens = [nltk_wtokeniser(i) for i in type_all]
        unique_tokens = flatten(tokens)
             
        # read data containing 10000 words

        f = pkgutil.get_data('mllibs',"corpus/wordlist.10000.txt")
        content = io.TextIOWrapper(io.BytesIO(f), encoding='utf-8')
        lines = content.readlines()
           
        cleaned = []
        for line in lines:
            removen = line.rstrip()
            if removen not in unique_tokens:
                cleaned.append(removen)
                
        corpus = typea + typeb + typec + typed + cleaned
        labels = [0,0,0,0,1,1,1,1,2,2,3,3,3] + [4 for i in range(len(cleaned))]
        data = pd.DataFrame({'corpus':corpus,
                             'label':labels})
        
        vectoriser = CountVectorizer(ngram_range=(1,2))
        X = vectoriser.fit_transform(data['corpus'])
        y = data['label'].values
        
        # we have a dissbalanced class model, so lets use class_weight

        model = DecisionTreeClassifier(class_weight={0:0.25,1:0.25,2:0.25,3:0.25,4:0.0001})
        model.fit(X,y)

        # with open('models/dtc_ner_tagger.pickle', 'wb') as f:
        #     pickle.dump(model, f)

        # with open('cv_ner_tagger.pickle', 'wb') as f:
        #     pickle.dump(vectoriser, f)

        # vectoriser_load = pkgutil.get_data('mllibs','models/cv_ner_tagger.pickle')
        # vectoriser = pickle.loads(vectoriser_load)

        # model_load = pkgutil.get_data('mllibs','models/dtc_ner_tagger.pickle')
        # model = pickle.loads(model_load)
        
        self.vectoriser['token_ner'] = vectoriser
        self.model['token_ner'] = model      
        self.label['token_ner'] = ['features','target','subset','data','other']             
        
    '''
    
    Model Predictions 
    
    '''
              
    # [sklearn] returns probability distribution (general)

    def test(self,name:str,command:str):
        test_phrase = [command]
        Xt_encode = self.vectoriser[name].transform(test_phrase)
        y_pred = self.model[name].predict_proba(Xt_encode)
        return y_pred

    # [sklearn] predict global task

    def predict_gtask(self,name:str,command:str):
        pred_per = self.test(name,command)     # percentage prediction for all classes
        val_pred = np.max(pred_per)            # highest probability value

        # (a) with decision threshold setting

        # if(val_pred > 0.5):
        #     idx_pred = np.argmax(pred_per)         # index of highest prob         
        #     pred_name = self.label[name][idx_pred] # get the name of the model class
        #     print(f"[note] found relevant global task [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
        # else:
        #     print(f'[note] no module passed decision threshold')
        #     pred_name = None

        # (b) without decision threshold setting

        idx_pred = np.argmax(pred_per)         # index of highest prob         
        pred_name = self.label[name][idx_pred] # get the name of the model class
        print(f"[note] found relevant global task [{pred_name}] w/ [{round(val_pred,2)}] certainty!")

        return pred_name,val_pred
    
    # [sklearn] predict module

    def predict_module(self,name:str,command:str):
        pred_per = self.test(name,command)     # percentage prediction for all classes
        val_pred = np.max(pred_per)            # highest probability value
        if(val_pred > 0.7):
            idx_pred = np.argmax(pred_per)         # index of highest prob         
            pred_name = self.label[name][idx_pred] # get the name of the model class
            print(f"[note] found relevant module [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
        else:
            print(f'[note] no module passed decision threshold')
            pred_name = None

        return pred_name,val_pred

    # [sklearn] predict task

    def predict_task(self,name:str,command:str):
        pred_per = self.test(name,command)     # percentage prediction for all classes
        val_pred = np.max(pred_per)            # highest probability value
        if(val_pred > 0.7):
            idx_pred = np.argmax(pred_per)                    # index of highest prob         
            pred_name = self.label[name][idx_pred] # get the name of the model class
            print(f"[note] found relevant activation function [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
        else:
            print(f'[note] no activation function passed decision threshold')
            pred_name = None

        return pred_name,val_pred
    
    # for testing only

    def dtest(self,corpus:str,command:str):
        
        print('available models')
        print(self.model.keys())
        
        prediction = self.test(corpus,command)[0]
        if(corpus in self.label):
            label = list(self.label[corpus])
        else:
            label = list(self.corpus_mt[corpus])
            
        df_pred = pd.DataFrame({'label':label,
                           'prediction':prediction})
        df_pred.sort_values(by='prediction',ascending=False,inplace=True)
        df_pred = df_pred.iloc[:5,:]
        display(df_pred)