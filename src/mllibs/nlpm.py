import pandas as pd
from mllibs.dict_helper import convert_dict_toXy,convert_dict_todf
from mllibs.tokenisers import PUNCTUATION_PATTERN
from mllibs.tokenisers import punktokeniser, custpunkttokeniser
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import clone
import itertools
import re
import numpy as np

def parse_json(json_data):

	'''
	
	Parse Module JSON files
	
	'''
  
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
	########################################################################
	
	Machine Learning Model Class
	
	self.ner_identifier [dict] contains all components of ner model
	
	########################################################################
	'''

	def __init__(self,modules):
		
		self.modules = modules # loaded modules
		
		# self.ner_identifier = {}
		self.gt = {}			  # global task classifier model storage
		self.sub_models = {}      # task label subset classifier models
	

	def create_gt_model(self,corpus:pd.DataFrame):
	
		X = corpus['text']
		y = corpus['task']

		# Create a pipeline with CountVectorizer and RandomForestClassifier
		pipeline = Pipeline([
			('vect', CountVectorizer(tokenizer=lambda x: custpunkttokeniser(x),
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

		self.gt['pipeline'] = pipeline	
		self.gt['labels'] = pipeline.named_steps['clf'].classes_

	def predict_gtask(self,command:str):
		pred_per = self.gt['pipeline'].predict_proba([command])
		val_pred = np.max(pred_per)
		idx_pred = np.argmax(pred_per)         # index of highest prob 
		pred_name = self.gt['labels'][idx_pred]
		self.gt['argmax'] = pred_name 
		self.gt['stats'] = pd.DataFrame({'classes':self.gt['labels'],
							'pp':list(pred_per[0])}).sort_values(by='pp',ascending=False)
      
		print(f"Found relevant global task [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
		
	# def predict_subtask(self,key:str,command:str):
	# 	pred_per = self.sub_models[key]['pipeline'].predict_proba([command])
	# 	val_pred = np.max(pred_per)
	# 	idx_pred = np.argmax(pred_per)         # index of highest prob 
	# 	pred_name = self.sub_models[key]['labels'][idx_pred]
	# 	self.sub_models[key]['argmax'] = pred_name 
	# 	self.sub_models[key]['stats'] = pd.DataFrame({'classes':self.sub_models[key]['labels'],'pp':list(pred_per[0])}).sort_values(by='pp',ascending=False)
		
	# 	print(f"Found relevant global task [{pred_name}] w/ [{round(val_pred,2)}] certainty!")
		
		
	def create_subset_model(self,corpus:pd.DataFrame):

		'''
		
		data : dict {'label':[corpus]} format 
		
		'''
		
		X = corpus['text']
		y = corpus['label']

		# vocabulary = ['-column','-list','-columns']
		# vocabulary.extend(self.modules.token_mparams) # ac accepted parameters
		# stop_words = ['a','the']

		# Create a pipeline with CountVectorizer and RandomForestClassifier
		pipeline = Pipeline([
			('vect', CountVectorizer(tokenizer=lambda x: x.split(),
					 				 ngram_range=(1,3))),
			('clf', GradientBoostingClassifier())
		])
		

		# Fit the pipeline on the training data
		pipeline.fit(X,y)
		y_pred = pipeline.predict(X)

		# Print classification report
		# print(classification_report(y, y_pred))
		return {'pipeline':pipeline,
		 		'labels': pipeline.named_steps['clf'].classes_}