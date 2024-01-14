from sklearn.preprocessing import LabelEncoder
from mllibs.tokenisers import nltk_wtokeniser,nltk_tokeniser, PUNCTUATION_PATTERN
from mllibs.nerparser import Parser,ner_model, tfidf, dicttransformer, merger_train, merger, ner_predict
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sklearn

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.optim as optim
import torch
import re

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
import os
import csv
import json
import requests
import zipfile
import time
# nltk.download('wordnet')

# import zipfile
# wordn = '/usr/share/nltk_data/corpora/wordnet.zip'
# wordnt = '/usr/share/nltk_data/corpora/'

# with zipfile.ZipFile(wordn,"r") as zip_ref:
#      zip_ref.extractall(wordnt)

# parse JSON module data

def parse_json(json_data):
  
    lst_classes = []; lst_corpus = []; lst_info = []; lst_corpus_sub = []
    for module in json_data['modules']:
        lst_corpus.append(module['corpus'])
        lst_corpus_sub.append(module['corpus_sub'])
        lst_classes.append(module['name'])
        lst_info.append(module['info'])
      
    return {'corpus':dict(zip(lst_classes,lst_corpus)),
            'corpus_sub':dict(zip(lst_classes,lst_corpus_sub)),
              'info':dict(zip(lst_classes,lst_info))}

# function to time exection time

def measure_execution_time(method):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {method.__name__}: {execution_time} seconds")
        return result
    return wrapper

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
        self.sub_models = {}      # task label subset classifier models

    @staticmethod
    def download_and_extract_zip(url, extract_path):

        # Send a GET request to the GitHub raw URL to download the ZIP file
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Create a file-like object from the downloaded content
            zip_file = io.BytesIO(response.content)
            
            # Extract the contents of the ZIP file to the specified extract path
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    '''
    ###########################################################################
    
    load module & prepare module content data
    
    ###########################################################################
    '''
    
    # helper function for subset model used in load
              
    @staticmethod
    def create_corpus_sub_model(data:dict):
      
      # convert a dict keys(labels), values(corpus documents) into X,y
      def convert_dict_toXy(data:dict):
        
        # Convert the dictionary to a list of tuples
        data_list = [(key, value) for key, values in data.items() for value in values]
        
        # Create a DataFrame from the list
        df = pd.DataFrame(data_list, columns=['label', 'text'])
        
        return df['text'],df['label']
      
      # prepare data
      X,y = convert_dict_toXy(data)
      
      # Create a pipeline with CountVectorizer and RandomForestClassifier
      pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=lambda x: x.split())),
        ('clf', RandomForestClassifier())
      ])
      
      # Fit the pipeline on the training data
      pipeline.fit(X,y)
      y_pred = pipeline.predict(X)
      
      # Print classification report
      print(classification_report(y, y_pred))
      return pipeline
    
    # group together all module data & construct corpuses
          
    def load(self,modules:list):
            
        def merge_dict_w_lists(data:dict):
          
          # Create a list of dictionaries
          list_of_dicts = [{key: values[i] if i < len(values) else None for key, values in data.items()} for i in range(max(map(len, data.values())))]
          
          # Create a dataframe from the list of dictionaries
          df = pd.DataFrame(list_of_dicts)
          return df
            
        print('[note] loading modules ...')
        
        # dictionary for storing model label (text not numeric)
        self.label = {} 
        
        # combined module information/option dictionaries
        
        lst_module_info = []
        lst_corpus = []
        dict_task_names = {}

        for module in modules:  
            
            # store module instance
            self.modules[module.name] = module
            
            '''
            
            Create subset pipelines for activation function labels (right away)
            
            '''
            
            # nested dict (for each label : subset corpus
            tdf_corpus_sub = module.nlp_config['corpus_sub']

            dict_data = {}; sub_models = {}
            for key,val in tdf_corpus_sub.items():
              if(type(val) is dict):
                self.sub_models[key] = self.create_corpus_sub_model(val)
  
            '''
            
            Prepare corpuses for activation functions
            
            '''
                
            # get dictionary with corpus
            tdf_corpus = module.nlp_config['corpus']
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
    
        def prepare_corpus(group:str) -> pd.DataFrame:
        
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
    
    BERT CLASSIFIER RELATED CLASSES 
    
    '''

    # Transformer based classification model
    # used for global activation function classification
    # requires [self.corpus_gt]

    # load global task transformer encoder classification model 
    # from stored model on github

    def load_trclassifier(self):

        # define corpus
        corpus = self.corpus_gt
        classes = len(corpus['class'].unique())

        '''

        Load the base models

        '''
        model_name = 'prajjwal1/bert-mini'
        # model_name = 'bert-base-uncased'
        tokeniser = BertTokenizer.from_pretrained(model_name)
        # model = BertForSequenceClassification.from_pretrained(model_name, 
        #                                                       num_labels=classes)

        # store tokeniser
        self.tokeniser['gt'] = tokeniser

        '''

        Read Fine-Tuned Classifier Model

        '''

        if(os.path.exists('local_classifier')):
            model = BertForSequenceClassification.from_pretrained('local_classifier',num_labels=classes)
            print('[note] using cached model(s)')
        else:
            print('[note] downloading model(s)')
            source = "https://github.com/mllibs/mllibs/raw/main/data/models/bert_classifier_model.zip" 
            self.download_and_extract_zip(source,'local_classifier')
            model = BertForSequenceClassification.from_pretrained('local_classifier',num_labels=classes)

        # store model
        self.model['gt'] = model

    # train global task transformer encoder classification model 
    # used on cloud only

    def train_trclassifier(self):

        # define corpus
        corpus = self.corpus_gt
        classes = len(corpus['class'].unique())

        # Load the pre-trained BERT model and tokenizer

        model_name = 'prajjwal1/bert-mini'
        # model_name = 'bert-base-uncased'

        tokeniser = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # modify later !
        le = LabelEncoder()
        targets = le.fit_transform(df['task'])
        data = {'corpus':list(df['text']),'labels':targets}

        # create dataset for text classification
        class CustomDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                return {'text': text, 'label': label}

        dataset = CustomDataset(list(data['corpus']),
                                list(data['labels']))

        # train bert model 
        def train_bert(dataset,tokeniser,model):

            # Define batch size and create data loader
            batch_size = 10
            dataloader = DataLoader(dataset, 
                                    sampler=RandomSampler(dataset), 
                                    batch_size=batch_size)

            # Set up optimizer and learning rate scheduler
            optimizer = AdamW(model.parameters(), lr=1e-5)
            criterion = nn.CrossEntropyLoss()
            total_steps = len(dataloader) * 2

            # Train the model
            model.train()
            for epoch in range(200):
                model.train()
                total_correct = 0
                total_samples = 0
                for batch in dataloader:
                
                    inputs = tokeniser(batch['text'], padding=True, truncation=True, return_tensors='pt')
                    inputs.to(device)
                    labels = batch['label'].to(device)

                    outputs = model(**inputs)
                    loss = criterion(outputs.logits, labels)
            
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Calculate accuracy
                predicted_labels = torch.argmax(outputs.logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            print(f'Epoch {epoch+1} completed. Accuracy: {accuracy:.4f}')    

        model = train_bert(dataset,tokeniser,model)

        if(device is 'gpu'):
            model.to('cpu')
    
        model.save_pretrained('bert_classifier_model')
        self.model['gt'] = model
        self.tokeniser['gt'] = tokeniser

    # RandomForest based classifier loop
    # Standard Random Forest + TF-IDF

    # @measure_execution_time
    def mlloop(self,corpus:dict,module_name:str):
        
        # Convert text to numeric representation         
        # vect = TfidfVectorizer(tokenizer=lambda x: nltk_wtokeniser(x))  
        stopwords = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS
        # vect = TfidfVectorizer(tokenizer=lambda x: nltk_wtokeniser(x),stop_words=['all','a','as','and']) 
        vect = CountVectorizer(tokenizer=lambda x: nltk_wtokeniser(x),stop_words=['all','a','as','and'])  
        vect.fit(corpus['text']) # input into vectoriser is a series
        vectors = vect.transform(corpus['text']) # sparse matrix
        self.vectoriser[module_name] = vect  # store vectoriser 

        # vocabulary of TFIDF
        lvocab = list(vect.vocabulary_.keys())
        lvocab.sort()
        self.vocabulary[module_name] = lvocab
        
        # X = np.asarray(vectors.todense())
        X = vectors
        y = corpus['class'].values.astype('int')

        model_rf = RandomForestClassifier()
        model = clone(model_rf)
        model.fit(X,y)
        self.model[module_name] = model # store model
        score = model.score(X,y)
        print(f"[note] training  [{module_name}] [{model}] [accuracy,{round(score,3)}]")

    # BERT based classifier loop
    # TF-IDF required to extract vocabulary

    def dlloop(self,corpus:dict,module_name:str):
        
        # Convert text to numeric representation         
        vect = TfidfVectorizer(tokenizer=lambda x: nltk_wtokeniser(x))
        vect.fit(corpus['text']) # input into vectoriser is a series
        vectors = vect.transform(corpus['text']) # sparse matrix
        self.vectoriser[module_name] = vect  # store vectoriser 

        # vocabulary of TFIDF
        lvocab = list(vect.vocabulary_.keys())
        lvocab.sort()
        self.vocabulary[module_name] = lvocab

        # preload model, store tokeniser,model
        self.load_trclassifier()

    # module selection model [ms]
    # > module class models [module name] x n modules
    
    def setup(self,type='mlloop'):

        self.vectoriser = {} # stores vectoriser
        self.model = {}      # storage for models
        self.tokeniser = {}  # store tokeniser 
        self.vocabulary = {} # vectoriser vocabulary
                    
        if(type == 'mlloop'):
            self.mlloop(self.corpus_gt,'gt')
            self.train_ner_tagger()
            print('[note] models trained!')
        if(type == 'load_bert'):
            self.dlloop(self.corpus_gt,'gt')
            self.train_ner_tagger()
            print('[note] model loaded!')
            
          
    '''
    ###########################################################################
    
    Prepare NER Model
    
    ###########################################################################
    '''

    # @measure_execution_time
    def train_ner_tagger(self):

        '''
        
        Load Models & Encoder
        
        '''

        # # f = pkgutil.get_data('mllibs', 'corpus/ner_modelparams_annot.csv')
        # path = pkg_resources.resource_filename('mllibs', '/corpus/ner_modelparams_annot.csv')
        # df = pd.read_csv(path,delimiter=',')
        # parser = Parser()
        # model,encoder = ner_model(parser,df)

        # self.ner_identifier['model'] = model
        # self.ner_identifier['encoder'] = encoder

        '''
        
        Train NER model
        
        '''

        parser = Parser()
        path = pkg_resources.resource_filename('mllibs', '/corpus/ner_corpus.csv')
        df = pd.read_csv(path,delimiter=',')

        def make_ner_corpus(parser,df:pd.DataFrame):

            # parse our NER tag data & tokenise our text
            lst_data = []; lst_tags = []
            for ii,row in df.iterrows():
                sentence = re.sub(PUNCTUATION_PATTERN, r" \1 ", row['question'])
                lst_data.extend(sentence.split())
                lst_tags.extend(parser(row["question"], row["annotated"]))
        
            return lst_data,lst_tags

        tokens,labels = make_ner_corpus(parser,df)
        # ldf = pd.DataFrame({'tokens':tokens,'labels':labels})

        X_vect1,tfidf_vectorizer = tfidf(tokens)            # imported function
        X_vect2,dict_vectorizer = dicttransformer(tokens)   # imported function
        X_all,model = merger_train(X_vect1,X_vect2,labels) # imported function
        # predict_label(X_all,tokens,labels,model)

        # self.ner_identifier['X_all'] = X_all
        self.ner_identifier['model'] = model
        self.ner_identifier['tfidf'] = tfidf_vectorizer
        self.ner_identifier['dict'] = dict_vectorizer

    def inference_ner_tagger(self,tokens:list):

        # ner classification model
        model = self.ner_identifier['model']

        # encoders
        tfidf_vectorizer = self.ner_identifier['tfidf']
        dict_vectorizer = self.ner_identifier['dict']

        X_vect1,_ = tfidf(tokens,tfidf_vectorizer)
        X_vect2,_ = dicttransformer(tokens,dict_vectorizer)
        X_all = merger(X_vect1,X_vect2)

        # store prediction
        self.ner_identifier['y_pred'] = ner_predict(X_all,tokens,model)

             
    '''
    
    Model Predictions 
    
    '''

    # [bert] inference

    def predict_gtask_bert(self,name:str,command:str):

        # label encoder should be identical to training labelencoder
        le = LabelEncoder()
        df = self.corpus_gt
        classes = len(df['class'].unique())
        targets = le.fit_transform(df['task'])

        model = self.model['gt'] 
        tokeniser = self.tokeniser['gt'] 

        # Tokenize the input text
        inputs = tokeniser(command, 
                           padding=True, 
                           truncation=True, 
                           return_tensors='pt')

        # Perform inference using the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted label
        predicted_label = torch.argmax(logits, dim=1).item()

        print(f"The predicted label for the input text is: {le.classes_[predicted_label]}")
        output = le.classes_[predicted_label]
            
        return output
              
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
        
    '''
    
    Remove Cached Models
    
    '''
      
    def reset_models(self):
      file_path = 'ner_catboost.bin'
      if os.path.exists(file_path):
        os.remove(file_path)
        print(f"[note] The file {file_path} has been successfully deleted.")
      else:
        print(f"[note] The file {file_path} does not exist.")