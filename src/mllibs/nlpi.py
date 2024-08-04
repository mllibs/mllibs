import pandas as pd
import numpy as np
import re
from mllibs.data_storage import data, check_data_compat
from mllibs.user_request import user_request
from mllibs.nlpm import nlpm
from mllibs.module import modules
from mllibs.modules.mpd_dfop import pd_dfop
from mllibs.modules.mstats_tests import stats_tests
from mllibs.modules.mstats_plot import stats_plot
import warnings; warnings.filterwarnings('ignore')


class nlpi:
	
	'''
	########################################################################

	Main Assembly Class 
	
	
	Accessible from other modules:
	
	self.modules
	
		.storage : list of module instances
		.info : module info from JSON
		.task_dict : {'module': {'function': ['corpus']}}
		.corpus_gt : corpus for global activation function classifier
	
	self.request
	
		find_neighbouring_tokens()
			
			.merged_tokens : find tokens separated by (and,[,])
		
		column_name_groupings()
		
			.grouped_col_names : group together df column names 
	
	########################################################################
	'''
	
	def __init__(self):
		
		# prepare modules & connections
		self.data = data()
		self.modules = modules()
		self.models = nlpm(self.modules)
		self.request = user_request(self.data,self.modules)
		self.test_mode = False
		self.module_args = {}  # dictionary passed to activation function
		self.iter = -1
		self.memory = {}
		
	# main user request
	def query(self,query_request:str):

		"""

		Main User Query Method 

		"""

		self.query_request = query_request # user request in string format
		self.parse_request()	# extract information from user request
		self.inference_request() # use query to predict activation function
		
		print('\n> Query information \n')
		print('> ',self.query_request)
		print('> ',' '.join(self.request.mtokens))


		if(self.test_mode == False):
			self.step_iteration()
		else:
			print('> [test mode] no iteration activated')



	def parse_request(self):

		"""
		
		Parse user request : Extract Data From Request
		
		"""
	
		self.request.store_tokens(self.query_request) 

		# self.data_in_tokens     # list : T/F for all tokens
		# self.column_in_tokens   # list : None/(data name)
		# self.merged_tokens
		# self.grouped_col_names

		# extract information from user request
		self.request.evaluate()

		# print(self.request.tokens)
		# print(self.request.data_in_tokens)
		# print(self.request.column_in_tokens)
		# print(self.request.grouped_token_idx)    # general and/, groupings
		
		# print(self.request.grouped_column_idx)  # column name groupings
		# print(self.request.grouped_column_names)
		
		# print(self.request.extracted_params)
		# print(self.request.extracted_column)
		# print(self.request.extracted_column_list)

		# store extracted data, parameters, columns
		self.module_args['data'] = self.request.extracted_data
		self.module_args['params'] = self.request.extracted_params  
		self.module_args['column'] = self.request.extracted_column
		self.module_args['column_list'] = self.request.extracted_column_list
		

	def inference_request(self):
		
		"""
		
		Global Activation Task Prediction 
		
		"""
		
		print('\n> Model predictions\n')

		# classification prediction of global activation function
		self.models.predict_gtask(' '.join(self.request.mtokens))

		# print(self.models.gt.keys())
		# print(self.models.gt['stats'])
		
		# predicted global activation function & its module
		self.pred_task = self.models.gt['argmax']
		self.pred_module = self.modules.info.loc[self.pred_task,'module']
		self.pred_info = self.modules.info.loc[self.pred_task].to_dict()

		self.module_args['pred_task'] = self.pred_task
		self.module_args['pred_info'] = self.pred_info
		
		"""
		
		Subset Prediction (if exists)
		
		"""
		
		if(self.pred_task in self.models.sub_models):
			
			pred_per = self.models.sub_models[self.pred_task]['pipeline'].predict_proba([' '.join(self.request.mtokens)])
			
			classes = self.models.sub_models[self.pred_task]['labels']
			val_pred = np.max(pred_per)            # highest probability value
			idx_pred = np.argmax(pred_per)         # index of highest prob         
			pred_name = classes[idx_pred] # get the name of the model class
			self.module_args['sub_task'] = pred_name
			print(f"Found sub_task [{pred_name}] w/ [{round(val_pred,2)}] certainty!")

		# print(self.request.extracted_data['storage_name'])
		# print(self.request.extracted_data['storage'])


	# train models used in nlpi
	def train_models(self):

		"""
		
		Train Models used in nlpi
		
		"""

		# train activation task classifier
		self.models.create_gt_model(self.modules.corpus_gt)
		
		# train subset models
		for key,corpus in self.modules.corpus_subset.items():
			self.models.sub_models[key] = self.models.create_subset_model(corpus)

	# pre-execution checks for activation function 
	def pre_iteration_checks(self):

		"""
		
		check if extracted data corresponds to ac input requirement
		
		"""

		self.check_data = check_data_compat(self.request.extracted_data['storage_name'],
						   		 	   		self.pred_info['data_compat'])
		

		"""
		
		check if extracted parameters, columns, data 
		(ie. format corresponds to activation function) requirements are met
		
		"""
		
		self.check_format = False

		# activation function acceptable formats 
		# can be dictionary : for subset or list (general)
		main_format = self.module_args['pred_info']['main_format']
		
		include_tokens = ['-column','-df','-columns','-list','-value','-range']
		for param in self.modules.token_mparams:
			include_tokens.append('~' + param)	

		# query compatibility; check keep only critical tags from request
		to_check_format = [i for i in self.request.mtokens if i in include_tokens]
		to_check_format.sort()

		# adjust to generalised format 
		case_id = to_check_format.count('-list')
		text = ' '.join(to_check_format)

		if(case_id > 2):
			to_check_format = re.sub(r'(-list\s+){2,}', '-mlist ', text)
		elif(case_id == 2):
			to_check_format = re.sub(r'(-list ){2}', '-dlist ', text)

		if(isinstance(main_format,dict)):

			# (a) subset defined in dictionary format

			# if subtask has been defined (it will be activated)
			# get the relevant subset 
			if(self.module_args['sub_task']):

				# subset_check_format -> list like main_format
				subset_check_format = main_format[self.module_args['sub_task']]

				# loop through and check compatibility
				for format in subset_check_format:
					lst_format = format.split(' ')
					lst_format.sort()
					str_format = ' '.join(lst_format)

					if(str_format == to_check_format):
						self.check_format = True
					
				if(self.check_format == False):
					print('> Format Problem!')
					print('\naccepted formats:')
					print(subset_check_format)
					print('\nprovided format:')
					print(to_check_format)
					
					

		elif(isinstance(main_format,list)):

			# (b) general activation function format

			for format in main_format:
				lst_format = format.split(' ')
				lst_format.sort()
				str_format = ' '.join(lst_format)

				if(str_format == to_check_format):
					self.check_format = True

			if(self.check_format == False):

				print('> Activation Function Format Problem!')
				print('\naccepted formats:')

				for case in main_format:
					tokens = case.split(' ')
					tokens.sort()
					print(' '.join(tokens))
				print('\nprovided format:')
				print(to_check_format)
				print('\n')
					
	
	# go through iteration	
	def step_iteration(self):
		
		self.iter += 1
		self.memory[self.iter] = {'data':None,'param':None,'column':None,'output':None}
		
		print(f'\n=== iteration {self.iter} ===========================\n')

		# pre iteration checks!
		self.pre_iteration_checks()

		if(self.check_data is True and self.check_format is True):

			result = self.modules.storage[self.pred_module].sel(self.module_args)
			if(result is not None):
				self.memory[self.iter]['result'] = result
				if(self.verbose):
					print(self.memory[self.iter]['result'])
		else:
			print('> pre iteration checks not passed')
			print(f"data: {self.check_data}, format: {self.check_format}")

		# reset storage data 
		self.request.reset_iter_storage()
		
	# add library modules (preset or list of instances)
	def add_modules(self,modules:list=None):

		"""
		
		Add Addition Modules 
		
		"""
		
		# preset modules
		if(modules is None):
			self.modules.load([pd_dfop(),
					  		   stats_tests(),
							   stats_plot()])
		else:
			self.modules.load(modules)	
	
		self.train_models()
	
	# add data to data sources 
	def add(self,data,name:str):
		self.data.add_data(data,name)

	# get the last result
	def glr(self):
		return self.memory[self.iter]['result']