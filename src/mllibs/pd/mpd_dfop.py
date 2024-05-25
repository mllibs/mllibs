from mllibs.nlpi import nlpi
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
from mllibs.dict_helper import print_dict
from mllibs.module_helper import get_single_element_value
import pkg_resources
import json
import numpy as np

'''

Data Exploration via Natural Language

'''

# sample module class structure
class pd_dfop(nlpi):

	def __init__(self):
		self.name = 'pd_dfop'
		path = pkg_resources.resource_filename('mllibs','/pd/mpd_dfop.json')
		with open(path, 'r') as f:
			self.json_data = json.load(f)
			self.nlp_config = parse_json(self.json_data)

	# set preset value from dictionary
	# if argument is already set

	# called in nlpi
	def sel(self,args:dict):

		self.select = args['pred_task']

		if(self.select == 'dfcolumninfo'):
			self.dfcolumninfo(args)
		if(self.select == 'dfsize'):
			self.dfsize(args)
		if(self.select == 'dfcolumn_distr'):

			'''

			Column value_counts

			'''

			ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
			if(args['sub_task'] == 'subset'):
				args['data'] = ldata[args['column']]
				self.dfcolumn_distr(args)
			elif(args['sub_task'] == 'all'):
				value_counts = self.value_counts_in_columns(ldata)
				nlpi.memory_output.append({'data':value_counts})

		if(self.select == 'dfna_all'):
			self.dfna_all(args)
		if(self.select == 'show_info'):
			self.show_info(args)
		if(self.select == 'show_dtypes'):
			self.show_dtypes(args)
		if(self.select == 'show_corr'):
			self.show_correlation(args)
		if(self.select == 'dfcolumn_unique'):

			'''
			
			Column unique values
			
			'''

			ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
			
			if(args['sub_task'] == 'all'):
				unique = self.unique_values_in_columns(ldata)
				print_dict(unique)
				nlpi.memory_output.append({'data':unique})
			elif(args['sub_task'] == 'one'):
				ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data'][args['column'][0]]
				print(ldata.unique())
			elif(args['sub_task'] == 'few'):
				ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data'][args['column']]
				unique = self.unique_values_in_columns(ldata)
				print_dict(unique)
				nlpi.memory_output.append({'data':unique})
				
		if(self.select == 'df_preview'):
			self.df_preview(args)
		if(self.select == 'show_stats'):
			self.show_statistics(args)
		if(self.select == 'dfna_column'):
			self.dfna_column(args)

		# convert column types

		# if(self.select == 'dfcolumn_tostr'):
		#     self.dfcolumn_tostr(args)
		# if(self.select == 'dfcolumn_toint'):
		#     self.dfcolumn_toint(args)
		# if(self.select == 'dfcolumn_dtype'):
		#     self.dfcolumn_dtype(args)

	'''

	ACTIVATION FUNCTIONS : ENTIRE DATAFRAME

	'''

	# show dataframe columns

	def dfcolumninfo(self,args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		print(ldata.columns)

	# show size of dataframe

	def dfsize(self,args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		print(ldata.shape)

	# show the dataframe information

	@staticmethod
	def show_info(args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		print(ldata.info())

	# show dataframe column data types

	@staticmethod
	def show_dtypes(args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		print(ldata.dtypes)

	# show numerical column linear correlation in dataframe

	@staticmethod
	def show_correlation(args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		numerical_df = ldata.select_dtypes(include=['number'])
		corr_mat = pd.DataFrame(np.round(numerical_df.corr(),2),
							 index = list(numerical_df.columns),
							 columns = list(numerical_df.columns))
		corr_mat = corr_mat.dropna(how='all',axis=0)
		corr_mat = corr_mat.dropna(how='all',axis=1)
		display(corr_mat)

	# show a preview of the dataframe

	@staticmethod
	def df_preview(args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		display(ldata.head())

	# show the missing data in all columns

	def dfna_all(self,args:dict):
		ldata = nlpi.data[get_single_element_value(args['data'],'df')]['data']
		print(ldata.isna().sum().sum(),'rows in total have missing data')
		print(ldata.isna().sum())

		print("[note] I've also stored the missing rows!")
		ls = args['data']
		nlpi.memory_output.append({'data':ldata[ldata.isna().any(axis=1)]})


	@staticmethod
	def unique_values_in_columns(df):

		'''

		Return unique values in all column of dataframe stored in dict

		'''

		unique_values = {}
		for col in df.columns:
			unique_values[col] = df[col].unique()
		return unique_values

	@staticmethod
	def value_counts_in_columns(df):

		'''

		Return the value_counts in all columns in a dataframe stored in a dict

		'''

		value_counts = {}
		for col in df.columns:
			value_counts[col] = df[col].value_counts().to_dict()
		print(value_counts)
		return value_counts



	'''

	ACTIVATION FUNCTIONS : SUBSETS OF DATAFRAME

	'''

	# column distribution for a single or multiple columns

	def dfcolumn_distr(self,args:dict):
		if(args['column'] != None):
			vals = args['data'][args['column']].value_counts(dropna=False)
			vals.name = 'values'
			perc = args['data'][args['column']].value_counts(dropna=False,normalize=True) * 100
			perc = perc.round(2)
			perc.name = 'percentage'
			display(pd.concat([vals,perc],axis=1))
		elif(args['col'] != None):
			vals = args['data'][args['col']].value_counts(dropna=False)
			vals.name = 'values'
			perc = args['data'][args['col']].value_counts(dropna=False,normalize=True) * 100
			perc = perc.round(2)
			perc.name = 'percentage'
			display(pd.concat([vals,perc],axis=1))
		else:
			print('[note] please specify the column name')

	# column unique values

	def dfcolumn_unique(self,args:dict):
		if(args['column'] == None and args['col'] == None):
			print('[note] please specify the column name')
		else:
			if(args['column'] != None):
				print(args['data'][args['column']].unique())
			elif(args['col'] != None):
				print(args['data'][args['col']].unique())

	# show the missing data in the column / if no column is provided show for all columns

	def dfna_column(self,args:dict):

		if(args['column'] != None):
			ls = args['data'][args['column']]
		elif(args['col'] != None):
			ls = args['data'][args['col']]
		else:
			print('[note] please specify the column name, showing for all columns')
			ls = args['data']

		if(isinstance(ls,pd.Series) == True):
			tls = ls.to_frame()
			print(tls.isna().sum().sum(),'rows in total have missing data')
			print("[note] I've stored the missing rows")
			idx = tls[tls.isna().any(axis=1)].index
			nlpi.memory_output.append({'data':args['data'].loc[idx]})
		elif(isinstance(ls,pd.DataFrame) == True):
			print(args['data'].isna().sum().sum(),'rows in total have missing data')
			print(args['data'].isna().sum())
			print("[note] I've stored the missing rows")
			nlpi.memory_output.append({'data':ls[ls.isna().any(axis=1)]})

	'''

	convert column types

	'''

	def dfcolumn_dtype(self,args:dict):

		# parameter dtype needs to have been set
		if(args['dtype'] != None):

			data = nlpi.data[args['data_name']]['data']
			column = args['column']

			try:
				data[column] = data[column].astype(args['dtype'])
				print('[note] modifying original dataset!')
			except:
				print(f"[note] can't modify the existing column type to {args['dtype']}")

	# convert column to string

	def dfcolumn_tostr(self,args:dict):

		data = nlpi.data[args['data_name']]['data']
		column = args['column']

		try:
			data[column] = data[column].astype('string')
			print('[note] modifying original dataset!')
		except:
			print("[note] can't modify the existing column type to string!")

	# convert column type to integer

	def dfcolumn_toint(self,args:dict):
		data = nlpi.data[args['data_name']]['data']
		column = args['column']

		try:
			data[column] = data[column].astype('int')
			print('[note] modifying original dataset!')
		except:
			print("[note] can't modify the existing column type to integer!")
			