import pandas as pd
import numpy as np
from src.data_storage import data, check_data_compat
from src.user_request import user_request
from src.nlpm import nlpm
from src.module import modules
from modules.mpd_dfop import pd_dfop
from modules.mstats_tests import stats_tests
from modules.mstats_plot import stats_plot

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
		
		# prepare modules & conmections
		self.data = data()
		self.modules = modules()
		self.models = nlpm(self.modules)
		self.request = user_request(self.data,self.modules)
		self.module_args = {}
		self.iter = -1
		self.memory = {}
		
	# main user request
	def query(self,query_request:str,verbose=False):
		self.verbose = verbose
		self.query_request = query_request # user request in string format
		self.parse_request()
		self.inference_request()
		
		print('\n> Query information \n')
		print('> ',self.query_request)
		print('> ',' '.join(self.request.mtokens))
		self.step_iteration()

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
		
		'''
		
		Global Activation Task Prediction 
		
		'''
		
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
		
		'''
		
		Subset Prediction (if exists)
		
		'''
		
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
		
		include_tokens = ['-column','-df','-columns','-list','-value']
		for param in self.modules.token_mparams:
			include_tokens.append('~' + param)	

		# query compatibility; check keep only critical tags from request
		to_check_format = [i for i in self.request.mtokens if i in include_tokens]
		to_check_format.sort()
		
		if(isinstance(main_format,dict)):

			# (a) subset defined in dictionary format

			# if subtask has been defined (it will be activated)
			# get the relevant subset 
			if(self.module_args['sub_task']):

				# subset_check_format -> list like main_format
				subset_check_format = main_format[self.module_args['sub_task']]

				for format in subset_check_format:
					lst_format = format.split(' ')
					lst_format.sort()

					if(lst_format == to_check_format):
						self.check_format = True

		elif(isinstance(main_format,list)):

			# (b) general activation function format

			for format in main_format:
				lst_format = format.split(' ')
				lst_format.sort()

				if(lst_format == to_check_format):
					self.check_format = True
					
	
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

	# get the last result
	def glr(self):
		return self.memory[self.iter]['result']
		
	# add library modules (preset or list of instances)
	def add_modules(self,modules:list=None):
		
		if(modules is None):
			self.modules.load([pd_dfop(),
					  		   stats_tests(),
							   stats_plot()])
		else:
			self.modules.load(modules)	
			
		#print(self.modules.corpus_gt)
		#print(self.modules.task_dict)	
	
		self.train_models()
	
	# add data to data sources 
	def add(self,data,name:str):
			self.data.add_data(data,name)





def main():
	
	iris = pd.read_csv('iris.csv')
	titanic = pd.read_csv('titanic.csv')

	i = nlpi() 
	i.add_modules() # add modules (create corpuses)
	i.add(iris,'iris') # add data to nlpi instance
	i.add(titanic,'titanic')

	sample1 = list(np.random.normal(scale=1, size=200))
	sample2 = list(np.random.rand(10))
	sample3 = list(np.random.exponential(scale=1, size=200))

	i.add(sample1,'sample1')
	i.add(sample3,'sample3')

	"""
	
	Module: Pandas DataFrame Operations
	
	"""

	# req = 'show a preview of the dataframe titanic'
	# req = 'describe the statistics of dataframe titanic'
	# req = 'show the shape of the dataframe titanic'
	# req = 'show the data types in the dataframe titanic'
	# req = 'show the columns that are present in the dataframe titanic'
	# req = 'show the correlation in titanic'

	# req = 'for all columns show the column distribution of for dataframe titanic'
	# req = 'for columns embarked and sex calculate the column distribution in titanic'
	# req = 'for column embarked calculate the column distribution in titanic'

	# req = 'for all the columns show the unique values in dataframe titanic'
	# req = 'for columns embarked and sex show the unique column values in titanic'
	# req = 'for the column embarked show the column unique values in the dataframe titanic'

	# req = 'show the missing data in the column embarked in the dataframe titanic'
	# req = 'show the missing data in the columns embarked and sex in titanic'
	# req = 'show the missing data in the dataframe titanic'

	"""
	
	Module : Statistical tests
	
	"""

	# req = 'evaluate the two sample independent students test using sample1 and sample3'
	# req = 'evaluate the dependent two sample ttest using samples sample1 and sample3'
	# req = 'one sample ttest on sample1 and compare to popmean of 0.12'
	#req = 'do a utest using sample1 and sample3'
	# req = "do the mann whitney utest for sample1 and sample3"

	# req = "check if the distribution follows a normal distribution using the kolmogorov smirnov test for sample1"
	# req = "having sample1 check if the distribution of the data follows a uniform distribution using the kolmogorov smirnov test"
	# req = "check if the data sample1 follows exponential distribution using the kolmogorov smirnov test"
	# req = "shapiro wilk test for normality for sample1"
	# req = "one way anova test on samples sample1 and sample3"
	
	"""
	
	Module : Statistical Plots
	
	"""
	
	# req = "show the histogram for samples sample1 and sample3"
	req = "show the histograms of samples sample1 sample3 set nbins = 30"
	req = "show the kernel density plot for samples sample1 and sample3"
	req = "show a visualisation of boxplots for samples sample1 and sample3"


	
	i.query(req)
	# print(i.glr())


if __name__ == '__main__':
	main()