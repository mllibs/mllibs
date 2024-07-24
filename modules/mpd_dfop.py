
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from src.nlpm import parse_json
import json
import os

def split_types(df:pd.DataFrame):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  
    numeric = df.select_dtypes(include=numerics)
    categorical = df.select_dtypes(exclude=numerics)
    return list(numeric.columns),list(categorical.columns)

class pd_dfop:
	
	def __init__(self):
		self.name = 'pd_dfop'
		path = 'modules/mpd_dfop.json'    
		with open(path, 'r') as f:
			self.json_data = json.load(f)
			self.nlp_config = parse_json(self.json_data)
		self.result = None

	def sel(self,args:dict):
		
		"""
		
		Activation Function Selection
		
		"""
		
		self.select = args['pred_task'] # activation function 
		self.data = args['data']['storage'] # stored data
		self.data_format = args['pred_info']['data_compat'] # input data forat
		self.columns = args['column_list']
		
		if(self.select == 'show_info'):
			self.show_info(args)	
		elif(self.select == 'df_describe'):
			self.df_describe(args)
		elif(self.select == 'df_preview'):
			self.df_preview(args)
		elif(self.select == 'df_shape'):
			self.df_shape(args)	
		elif(self.select == 'df_type'):
			self.df_type(args)	
		elif(self.select == 'df_columns'):
			self.df_columns(args)	
		elif(self.select == 'df_corr'):
			self.df_corr(args)	
			
		elif(self.select == 'dfcolumn_distr'):

			# column distribution (categorical only)

			if(args['sub_task'] == 'one'):
				self.df_valuecount_one(args)				
			if(args['sub_task'] == 'multiple'):
				self.df_valuecount_multi(args)
			elif(args['sub_task'] == 'all'):
				self.df_valuecount_all(args)

		elif(self.select == 'dfcolumn_unique'):

			# unique column values in dataframe

			if(args['sub_task'] == 'one'):
				self.df_unique_one(args)				
			if(args['sub_task'] == 'multiple'):
				self.df_unique_multi(args)
			elif(args['sub_task'] == 'all'):
				self.df_unique_all(args)

		elif(self.select == 'df_na'):

			# missing data in dataframe
			if(args['sub_task'] == 'one'):
				self.df_na_one(args)				
			if(args['sub_task'] == 'multiple'):
				self.df_na_multi(args)
			elif(args['sub_task'] == 'all'):
				self.df_na_all(args)

		return self.result
		
	"""
	=========================================================

	Activation Functions
	
	=========================================================
	"""
	
	def show_info(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			ldf.info()
			
	def df_describe(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			self.result = ldf.describe()
			
	def df_preview(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			self.result = ldf.head()
			print(self.result)
			
	def df_shape(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			self.result = ldf.shape
			
	def df_type(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			self.result = ldf.dtypes
		
	def df_columns(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			self.result = list(ldf.columns)
			
	def df_corr(self,args:dict):

		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			numeric_columns,categorical_columns = split_types(ldf)
			tdf = ldf[numeric_columns]
			self.result = tdf.corr().round(3)

	"""
	
					Missing Data (subtasks)

	======================================================
	"""
			
	def df_na_one(self,args:dict):
		
		'''
		
		self.columns should be format [a]
		
		'''
		
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0][self.columns[0]]
			type_id = ldf.dtype
			if(type_id != float):
				self.result = ldf.isna().sum()

	def df_na_multi(self,args:dict):
		
		'''
		
		self.columns should be format [[a,b,c]]
		
		'''

		# for each column save the value counts 
		column_data = {}
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0][self.columns[0]] # dataframe
			columns = list(ldf.columns)
			self.result = ldf[columns].isna().sum()
			# for col_name in columns:
			# 	type_id = ldf[col_name].dtype
			# 	if(type_id != float):
			# 		column_data[col_name] = ldf[col_name].isna().sum()
			# self.result = column_data


	def df_na_all(self,args:dict):
		
		'''
		
		no columns are references; all non floats selected
		
		'''
		
		# for each column save the value counts 
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			self.result = ldf.isna().sum()
			
	"""
	
					Value Counts (subtasks)

	======================================================
	"""

	def df_valuecount_all(self,args:dict):
		
		'''
		
		no columns are references; all non floats selected
		
		'''
		
		# for each column save the value counts 
		column_data = {}
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			columns = list(ldf.columns)
			for col_name in columns:
				type_id = ldf[col_name].dtype
				if(type_id != float):
					column_data[col_name] = ldf[col_name].value_counts()
			self.result = column_data
			
	def df_valuecount_multi(self,args:dict):
		
		'''
		
		self.columns should be format [[a,b,c]]
		
		'''
		
		# for each column save the value counts 
		column_data = {}
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0][self.columns[0]] # dataframe
			columns = list(ldf.columns)
			for col_name in columns:
				type_id = ldf[col_name].dtype
				if(type_id != float):
					column_data[col_name] = ldf[col_name].value_counts()
			self.result = column_data
	
	def df_valuecount_one(self,args:dict):
		
		'''
		
		self.columns should be format [a]
		
		'''
		
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0][self.columns[0]]
			type_id = ldf.dtype
			if(type_id != float):
				self.result = ldf.value_counts()


	"""
	
					Unique Values (subtasks)

	======================================================
	"""

	def df_unique_all(self,args:dict):
		
		'''
		
		no columns are references; all non floats selected
		
		'''
		
		# for each column save the value counts 
		column_data = {}
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0] 
			columns = list(ldf.columns)
			for col_name in columns:
				type_id = ldf[col_name].dtype
				if(type_id != float):
					column_data[col_name] = ldf[col_name].unique()
			self.result = column_data
			
	def df_unique_multi(self,args:dict):
		
		'''
		
		self.columns should be format [[a,b,c]]
		
		'''
		
		# for each column save the value counts 
		column_data = {}
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0][self.columns[0]] # dataframe
			columns = list(ldf.columns)
			for col_name in columns:
				type_id = ldf[col_name].dtype
				if(type_id != float):
					column_data[col_name] = ldf[col_name].unique()
			self.result = column_data
	
	def df_unique_one(self,args:dict):
		
		'''
		
		self.columns should be format [a]
		
		'''
		
		# check data compatibility, use one dataframe
		if(self.data_format == 'df'):
			ldf = self.data['df'][0][self.columns[0]]
			type_id = ldf.dtype
			if(type_id != float):
				self.result = ldf.unique()
