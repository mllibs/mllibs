from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel,sigmoid_kernel
from sklearn.base import clone
from collections import OrderedDict
import numpy as np
import pandas as pd
import zipfile
import pkgutil
import io

import nltk
nltk.download('wordnet')

wordn = '/usr/share/nltk_data/corpora/wordnet.zip'
wordnt = '/usr/share/nltk_data/corpora/'

with zipfile.ZipFile(wordn,"r") as zip_ref:
     zip_ref.extractall(wordnt)

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer


'''


NLPM class

- combine together all extension modules
- create machine learning models for task prediction



'''


class nlpm:
    
    def __init__(self):
        self.task_dict = {} # stores the input task variation dictionary (prepare)
        self.functions = {} # stores model associate function class (prepare) 
        self.model_class = {}
        
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
    
    # group together all module data & construct corpuses
            
    def load(self,modules:list):
            
        if(type(modules) is list):
            
            print('loading modules ...')
            
            # dictionary for storing model label (text not numeric)
            self.label = {} 
            
            # combined module information/option dictionaries
            
            lst_module_info = []; lst_corpus = []; dict_task_names = {}
            for module in modules:  
                
                # store module functions
                self.functions[module.name] = module
                
                # combine corpuses of modules
                tdf_corpus = module.nlp_config['corpus'] # get dictionary with corpus
                df_corpus = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_corpus.items()]))          
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
            
            # module order for ms
            self.mod_order = unique_module_groupby
            
            ''' Step 4 : labels for other models (based on provided info) '''
            
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
            
               
    
        else:
            raise TypeError('please make input a list of modules!')
            
    ''' 
    
    MACHINE LEARNING LOOP 
    
    '''
    
    # countvectoriser
    # logistic regression

    
    def mlloop(self,corpus,module_name):
        
        ''' Preprocess text data '''
        
        # vect = CountVectorizer()
#        vect = CountVectorizer(tokenizer=lambda x: word_tokenize(x))
        vect = CountVectorizer(tokenizer=lambda x: WhitespaceTokenizer().tokenize(x))
#         lvect = clone(vect)
        
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
                    
        if(type == 'mlloop'):
        
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
    
            self.toksub_model()
            self.ner_tokentag_model()

            print('models trained...')
        
    '''
    ///////////////////////////////////////////////////////////////
    
    ADDITIONAL MODELS
    
    ///////////////////////////////////////////////////////////////
    
    ''' 
    
    ''' 
    
    1. CREATE SUBSET DETERMINATION MODEL 
    
    create multiclass classification model which will determine 
    which approach to utilise for the selection of subset features    
    
    '''
     
    def toksub_model(self):
        
       
        # corpus
        lst_data = ['select all but','all features except for','all features except',
                    'select all features except for','all but','all except','all columns but',
            
                    'features','with features','select features','select only','select','only columns',
                    
                    'imported list','from imported data','select features from list',
                    'from imported list','select data from list','columns from data',
                    
                    'numerical features only','numeric features','values','numerical features','values only',
                    
                    'categorical features only','categorical','select categorical','categorical only','categoric',

        'select all', 'get all','choose all','pick all']
                    
                    
        # [0] select all but X
        # [1] select only X
        # [2] select from list
        # [3] select numeric only
        # [4] select categorical
        
        lst_tag = [0,0,0,0,0,0,0, 
                   1,1,1,1,1,1,
                   2,2,2,2,2,2,
                   3,3,3,3,3,
                   4,4,4,4,4,
                   5,5,5,5]
                  
                  
        data = pd.DataFrame({'corpus':lst_data,'label':lst_tag})

        vectoriser = CountVectorizer(stop_words=['using','use'])
        X = vectoriser.fit_transform(list(data['corpus'])).toarray()
        y = data['label'].values

        model = LogisticRegression().fit(X,y)
        
        self.vectoriser['token_subset'] = vectoriser
        self.model['token_subset'] = model      
        self.label['token_subset'] = ['allbut','only','fromdata','numeric','categorical','all']

    '''
    //////////////////////////////////////////////////////////////////
    
    2. NER sentence splitting model 
    identify key tokens; features, target, subset & data

    //////////////////////////////////////////////////////////////////                
    '''
        
    def ner_tokentag_model(self):

        flatten = lambda l: [item for sublist in l for item in sublist]
        
        def tokenise(text):
            return WhitespaceTokenizer().tokenize(text)
        
        typea = ['features','feature list','feature columns','independent']
        typeb = ['target','target column','target variable','dependent']
        typec = ['subset','subset columns']
        typed = ['data','data source','source']
        type_all = typea + typeb + typec + typed
        tokens = [tokenise(i) for i in type_all]
        unique_tokens = flatten(tokens)
        
        #with open('corpus/wordlist.10000.txt') as f:
            #lines = f.readlines()
             
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
        data = pd.DataFrame({'corpus':corpus,'label':labels})
        
        vectoriser = CountVectorizer(ngram_range=(1,2))
        X = vectoriser.fit_transform(data['corpus']).toarray()
        y = data['label'].values
        
        # we have a dissbalanced class model, so lets use class_weight
        model = DecisionTreeClassifier(class_weight={0:0.25,1:0.25,2:0.25,3:0.25,4:0.0001})
        model.fit(X,y)
        model.predict(X)
        
        self.vectoriser['token_ner'] = vectoriser
        self.model['token_ner'] = model      
        self.label['token_ner'] = ['features','target','subset','data','other']             
        
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
        pred_name = self.label[name][idx_pred]  # get the name of the model class
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