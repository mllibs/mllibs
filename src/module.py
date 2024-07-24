
import pandas as pd
from ner_parser import Parser, dicttransformer, tfidf, merger
from src.dict_helper import convert_dict_toXy,convert_dict_todf
from src.tokenisers import PUNCTUATION_PATTERN
from src.tokenisers import punktokeniser, custpunkttokeniser
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
import itertools
import re
import numpy as np

class modules:
	
	'''
	########################################################################
	
	Custom Modules Assembly Class
	
	self.storage : list of module instances
	self.info : module info from JSON
	self.task_dict : {'module': {'function': ['corpus']}}
	self.corpus_gt : corpus for global activation function classifier
	self.token_mparams : preset module parameters
	
	########################################################################
	'''

	def __init__(self):
		self.storage = {}
		self.task_dict = {}
		self.label = {} #  storing model label (text not numeric)
		
	def load(self,modules:list):
			
		def merge_dict_w_lists(data:dict):
	
			# Create a list of dictionaries
			list_of_dicts = [{key: values[i] 
                                if i < len(values) 
                                else None 
                                    for key, values in data.items()} 
                                    for i in range(max(map(len, data.values())))]
			df = pd.DataFrame(list_of_dicts)
			return df
			
		print('[note] loading modules ...')
	
		
		# combined module information/option dictionaries
		
		lst_module_info = []
		lst_corpus = []
		dict_task_names = {}
		self.corpus_subset = {}	

		for module in modules:  
			
			# store module instance
			self.storage[module.name] = module

			'''
			
			Prepare corpuses for activation functions, models trained later
			
			'''
				
			# get dictionary with corpus
			tdf_corpus = module.nlp_config['corpus']
			df_corpus = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_corpus.items()]))
			  
			# module task list
			dict_task_names[module.name] = list(df_corpus.columns)  

			lst_corpus.append(df_corpus)
			self.task_dict[module.name] = tdf_corpus     # save corpus
			
			# combine info of different modules
			opt = module.nlp_config['info']     # already defined task corpus
			tdf_opt = pd.DataFrame(opt)
			df_opt = pd.DataFrame(dict([(key,pd.Series(value)) for key,value in tdf_opt.items()]))
			lst_module_info.append(df_opt)
			

		# update label dictionary to include loaded modules
		self.label.update(dict_task_names)  

		# create task corpuses
		corpus = pd.concat(lst_corpus,axis=1)
		

		for module in modules: 
		
			'''
		
			Create corpus for subset
		
			'''
			
			# store module instance
			self.storage[module.name] = module

			# nested dict (for each label : subset corpus
			tdf_corpus_sub = module.nlp_config['corpus_sub']

			# key - module name
			# value - activation function's subset label & subset corpus

			for key,val in tdf_corpus_sub.items():
				if(type(val) is dict):
					ldf = convert_dict_todf(val)
					self.corpus_subset[key] = ldf  # save for future reference

		'''
		
		Extract unique input parameters that one can use in a user request
		
		'''

		lst_temp = []
		for module in modules:
			for af,val in module.nlp_config['info'].items():
				if(module.nlp_config['info'][af]['arg_compat'] != 'None'):
					lst_temp.extend(module.nlp_config['info'][af]['arg_compat'].split())
		
		# module parameter tokens
		self.token_mparams = list(set(" ".join(lst_temp).split(' ')))
		
		# task information options
		combined_opt = pd.concat(lst_module_info,axis=1)
		combined_opt = combined_opt.T.sort_values(by='module')
		combined_opt_index = combined_opt.index

		# Create Module Corpus Labels

		# note groupby (alphabetically module order) (module order setter)
		module_groupby = dict(tuple(combined_opt.groupby(by='module')))
		
		for i in module_groupby.keys():
			ldata = module_groupby[i]
			ldata['task_id'] = range(0,ldata.shape[0])

		df_opt = pd.concat(module_groupby).reset_index(drop=True)
		df_opt.index = combined_opt_index

		# generate task labels    
		encoder = LabelEncoder()
		df_opt['gtask_id'] = range(df_opt.shape[0])
		self.label['gt'] = list(combined_opt_index)
		
		# JSON Module > Info summary
		self.info = df_opt
			
		# Create Global Task Selection Corpus

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
		# self.corpus_sub = prepare_corpus('subtopic_id') # subtopic tasks