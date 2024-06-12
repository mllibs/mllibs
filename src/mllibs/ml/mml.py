from mllibs.module_helper import confim_dtype, get_ddata, get_mdata, get_sdata, get_dmdata, get_spdata, get_nested_list_and_indices
from sklearn.linear_model import LogisticRegression, LinearRegression
from mllibs.nlpi import nlpi
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json

'''

train scikit-learn models

'''

class scikitml(nlpi):
	
	def __init__(self):
		self.name = 'ml'  

		path = pkg_resources.resource_filename('mllibs','/ml/mml.json')
		with open(path, 'r') as f:
			self.json_data = json.load(f)
			self.nlp_config = parse_json(self.json_data)
		
	# select activation function
	def sel(self,args:dict):

		select = args['pred_task']
		self.data = args['data']
		self.info = args['task_info']['description']
		self.sub_task = args['sub_task']
		self.column = args['column']

		def get_data(dtype_id:str,case_id:str,names_id=False):

			'''

				Get the data (names_id == False) or just return names if (names_id == True)

			'''

			if(case_id == 'sdata'):
				ldata_names = get_sdata(args['data'],dtype_id)
			elif(case_id == 'ddata'):
				ldata_names = get_ddata(args['data'],dtype_id)
			elif(case_id == 'mdata'):
				ldata_names = get_mdata(args['data'],dtype_id)
			elif(case_id == 'dmdata'):
				ldata_names = get_dmdata(args['data'],dtype_id)
			elif(case_id == 'spdata'):
				ldata_names = get_spdata(args['data'],dtype_id)

			self.data_name = ldata_names

			# return data or names only
			if(names_id == False):
				if(ldata_names != None):
					if(len(ldata_names) != 1):
						ldata = []
						for dname in ldata_names:
							ldata.append(nlpi.data[dname]['data'])
						return ldata
					elif(len(ldata_names) == 1):
						return nlpi.data[ldata_names[0]]['data']
				else:
					return None 
			else:
				if(ldata_names != None):
					return ldata_names
				else:
					return None
		
		if(select == 'mlop_logreg'):

			# get single dataframe
			args['data'] = get_data('df','sdata')
			if(args['data'] is not None):

				# columns w/o parameter treatment
				if(self.column != None):
					group_col_idx,indiv_col_idx = get_nested_list_and_indices(self.column)

					# group column names
					group_col = self.column[group_col_idx]

					# non grouped column names
					lst_indiv = []
					for idx in indiv_col_idx:
						lst_indiv.append(self.column[idx])

			self.column = group_col

			self.fit_logreg(args)

		if(select == 'mlop_linreg'):

			# get single dataframe
			args['data'] = get_data('df','sdata')
			if(args['data'] is not None):

				# columns w/o parameter treatment
				if(self.column != None):
					group_col_idx,indiv_col_idx = get_nested_list_and_indices(self.column)

					# group column names
					group_col = self.column[group_col_idx]

					# non grouped column names
					lst_indiv = []
					for idx in indiv_col_idx:
						lst_indiv.append(self.column[idx])

			self.column = group_col

			self.fit_linreg(args)

	'''

	Activation Functions

	'''

	# fit a logistic regression model
	def fit_logreg(self,args:dict) -> None:

		model = LogisticRegression()

		target_variable = nlpi.data[self.data_name[0]]['target']
		if(target_variable is not None):
			y = args['data'][target_variable].copy()
			X = args['data'][self.column].copy()

			if(X.isna().sum().sum() < 1):

				model.fit(X,y)
				nlpi.memory_output[nlpi.iter] = {'features':X,'target':y,'model':model}

			else:
				print('[note] data contains missing data')

		else:
			print('[note] target variable not set')

	# fit a logistic regression model
	def fit_linreg(self,args:dict) -> None:

		model = LinearRegression()

		target_variable = nlpi.data[self.data_name[0]]['target']
		if(target_variable is not None):
			y = args['data'][target_variable].copy()
			X = args['data'][self.column].copy()

			if(X.isna().sum().sum() < 1):

				model.fit(X,y)
				nlpi.memory_output[nlpi.iter] = {'features':X,'target':y,'model':model}

			else:
				print('[note] data contains missing data')

		else:
			print('[note] target variable not set')

