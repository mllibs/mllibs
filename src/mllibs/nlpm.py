from sklearn.preprocessing import LabelEncoder
from mllibs.tokenisers import nltk_wtokeniser,nltk_tokeniser, PUNCTUATION_PATTERN
from mllibs.nerparser import Parser,ner_model, tfidf, dicttransformer, merger
from mllibs.dict_helper import convert_dict_toXy,convert_dict_todf
from catboost import CatBoostClassifier
import itertools

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report
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
import re


def parse_json(json_data):
  
	lst_classes = []; lst_corpus = []; lst_info = []; lst_corpus_sub = []
	
	for module in json_data['modules']:

		'''

			Make Activation Function Classifier Corpus
		
		'''
		# if "subset" get all the data from the subset dictionary
		# and use it to make the activation function corpus
		
		if(module['corpus'] == "subset"):
			temp_corpus = []
			for _,value in module['corpus_sub'].items():
				temp_corpus.extend(value)
			lst_corpus.append(temp_corpus)
		else:
			lst_corpus.append(module['corpus'])

		lst_corpus_sub.append(module['corpus_sub'])
		lst_classes.append(module['name'])
		lst_info.append(module['info'])
	  
	return {'corpus':dict(zip(lst_classes,lst_corpus)),
			'corpus_sub':dict(zip(lst_classes,lst_corpus_sub)),
			  'info':dict(zip(lst_classes,lst_info))}


class nlpm:


	'''
	###########################################################################

				NLPM class

				> Combine together all extension modules
				> Create machine learning models for task prediction

	###########################################################################
				

		create_corpus_sub_model  : sub_task model (requires X,y)



	'''
	
	def __init__(self):
		print('\n[note] initialising nlpm, please load modules using .load(list)')
		self.task_dict = {} # stores the input task variation dictionary (prepare)
		self.modules = {} # stores model associate function class (prepare) 
		self.ner_identifier = {}  # NER tagger (inactive)
		self.sub_models = {}      # task label subset classifier models

	'''
	###########################################################################
	
	load module & prepare module content data
	
	###########################################################################
	'''
	
	# helper function for subset model used in load
			  
	def create_corpus_sub_model(self,X:pd.Series,y:pd.Series):

		'''
		
		data : dict {'label':[corpus]} format 
		
		'''

		vocabulary = ['-column','-list']
		vocabulary.extend(self.token_mparams)
		# stop_words = ['a','the']

		# Create a pipeline with CountVectorizer and RandomForestClassifier
		pipeline = Pipeline([
			('vect', CountVectorizer(tokenizer=lambda x: x.split(),ngram_range=(1,1),stop_words=['a','the'],vocabulary=vocabulary)),
			('clf', GradientBoostingClassifier())
		])


		# Fit the pipeline on the training data
		pipeline.fit(X,y)
		y_pred = pipeline.predict(X)

		# Print classification report
		# print(classification_report(y, y_pred))
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
		self.corpus_subset = {}	

		for module in modules:  
			
			# store module instance
			self.modules[module.name] = module

			'''
			
			Prepare corpuses for activation functions, models trained later
			
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
		
		Extract unique input parameters that one can use in a user request
		
		'''

		lst_temp = []
		for module in modules:
			for af,val in module.nlp_config['info'].items():
				if(module.nlp_config['info'][af]['arg_compat'] != 'None'):
					lst_temp.append(module.nlp_config['info'][af]['arg_compat'])

		lst_new = []
		for i in lst_temp:
			lst_new.append('~' + i)
		
		self.token_mparams = list(set(" ".join(lst_new).split(' ')))

			
		'''

		Train Model for Subset 

		'''

		for module in modules:  
			
			# store module instance
			self.modules[module.name] = module

			# nested dict (for each label : subset corpus
			tdf_corpus_sub = module.nlp_config['corpus_sub']

			# key - module name
			# value - activation function's subset label & subset corpus

			for key,val in tdf_corpus_sub.items():
				if(type(val) is dict):
					ldf = convert_dict_todf(val)
					self.corpus_subset[key] = ldf  # save for future reference
					X,y = convert_dict_toXy(val)  # prepare X,y for model
					self.sub_models[key] = self.create_corpus_sub_model(X,y)



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

		# self.corpus_ms = prepare_corpus('module_id') # modue selection dataframe
		self.corpus_gt = prepare_corpus('gtask_id')  # global task dataframe
		# self.corpus_act = prepare_corpus('action_id') # action task dataframe
		# self.corpus_top = prepare_corpus('topic_id') # topic task dataframe
		# self.corpus_sub = prepare_corpus('subtopic_id') # subtopic tasks dataframe

		

	'''

				RandomForest based classifier loop
				Standard Random Forest + TF-IDF

	'''



	def mlloop(self,corpus:pd.DataFrame,module_name:str):

		X = corpus['text']
		y = corpus['class']

		# Create a pipeline with CountVectorizer and RandomForestClassifier
		pipeline = Pipeline([
			('vect', CountVectorizer(tokenizer=lambda x: nltk_wtokeniser(x),
									 ngram_range=(1,1),
									 stop_words=['all','a','as','and'])),
			('clf', RandomForestClassifier())
		])

		# Fit the pipeline on the training data
		pipeline.fit(X,y)
		y_pred = pipeline.predict(X)

		# Print classification report
		# print(classification_report(y, y_pred))
		score = pipeline.score(X,y)
		print(f"[note] training  [gt_model] [accuracy,{round(score,3)}]")

		self.gt = pipeline



	'''

	module selection model [ms]
	module class models [module name] x n modules
	
	'''

	def setup(self,type='mlloop'):   
	
		if(type == 'mlloop'):
			self.mlloop(self.corpus_gt,'gt')

#			self.ner_identifier['model'] = model
#			self.ner_identifier['tfidf'] = tfidf_vectorizer#
#			self.ner_identifier['dict'] = dict_vectorizer

			self.train_ner_tagger()
			print('[note] models trained!')
			
		  
	'''
	###########################################################################
	
								Prepare NER Model
	
	###########################################################################
	'''

	def train_ner_tagger(self):

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

		# convert to non-sparse 
		X_vect1 = pd.DataFrame(np.asarray(X_vect1.todense()))
		X_vect2 = pd.DataFrame(np.asarray(X_vect2.todense()))
		data = pd.concat([X_vect1,X_vect2],axis=1)
		data.fillna(0.0,inplace=True)

		self.tdata = data
		self.tfidf = tfidf_vectorizer
		self.dictv = dict_vectorizer
		self.vlabels = labels

		data = data.values
    
		# try:
		#     # Load the model from the file
		#     model = CatBoostClassifier(silent=True)
		#     model.load_model('ner_catboost.bin')
		#     print('[note] using cached ner catboost model')
		# except:
		# model = CatBoostClassifier(silent=True)
		# model.fit(data,labels)
		# model.save_model('ner_catboost.bin')

		model = RandomForestClassifier()
		model.fit(data,labels)

		# self.ner_identifier['X_all'] = X_all
		self.ner_identifier['model'] = model
		self.ner_identifier['tfidf'] = tfidf_vectorizer
		self.ner_identifier['dict'] = dict_vectorizer

	def inference_ner_tagger(self,tokens:list):

		'''
		
			Inference on User Request : Method for classifying (NER) a list of strings
		
		'''

		# ner classification model
		model = self.ner_identifier['model']

		# encoders
		tfidf_vectorizer = self.ner_identifier['tfidf']
		dict_vectorizer = self.ner_identifier['dict']

		X_vect1,_ = tfidf(tokens,tfidf_vectorizer)
		X_vect2,_ = dicttransformer(tokens,dict_vectorizer)
		X_all = merger(X_vect1,X_vect2)

		# predict
		y_pred = model.predict(X_all)

		# print(list(itertools.chain(*y_pred)))
		# self.ner_identifier['y_pred'] = list(itertools.chain(*y_pred))
		self.ner_identifier['y_pred'] = y_pred
		#display(pd.DataFrame({'y':tokens,
		#                      'yp':list(itertools.chain(*y_pred))}).T)
			 
	'''
	
	Model Predictions 
	
	'''

	# [sklearn] predict global task

	def predict_gtask(self,name:str,command:str):
		pred_per = self.gt.predict_proba([command])
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

		self.gt_stats = pd.DataFrame({'classes':list(self.gt.classes_),
						'pp':list(pred_per[0])}).sort_values(by='pp',ascending=False)
		
		mapper = {index: value for index, value in enumerate(self.label['gt'])}
		self.gt_stats['classes'] = self.gt_stats['classes'].map(mapper)


		return pred_name,val_pred
	
	# for testing only

	# def dtest(self,corpus:str,command:str):
		
	# 	print('available models')
	# 	print(self.model.keys())
		
	# 	prediction = self.test(corpus,command)[0]
	# 	if(corpus in self.label):
	# 		label = list(self.label[corpus])
	# 	else:
	# 		label = list(self.corpus_mt[corpus])
			
	# 	df_pred = pd.DataFrame({'label':label,
	# 					   'prediction':prediction})
	# 	df_pred.sort_values(by='prediction',ascending=False,inplace=True)
	# 	df_pred = df_pred.iloc[:5,:]
	# 	display(df_pred)
		