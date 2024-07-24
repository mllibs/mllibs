import pandas as pd


class data:
	
	'''
	########################################################################

	Data Sources Class

	########################################################################
	'''
     
	def __init__(self):
		self.storage = {}		  # data_name : data
		self.dtype = {}		  # data_name : data type
		self.column_names = {}
		self.column_types = {}
		self.column_statistics = {}
		self.target = {}
		self.active_columns = {}
		

	def add_data(self,data:list,data_name:str):

		if(isinstance(data,list)):
			self.storage[data_name] = data
			self.dtype[data_name] = 'list'   

		elif(isinstance(data,pd.DataFrame)):
			self.storage[data_name] = data
			self.dtype[data_name] = 'df'
			self.column_names[data_name] = self.get_pdf_colnames(data_name)
			self.column_types[data_name] = self.get_pdf_coldtype(data_name)
			self.column_statistics[data_name] = self.get_pdf_colstatistics(data_name)
			self.target[data_name] = None
			self.active_columns[data_name] = {}
		else:
			print('[note] data is not dataframe')

	def get_pdf_colnames(self,data_name:str):
		return self.storage[data_name].columns.tolist()

	def get_pdf_coldtype(self,data_name:str):
		return self.storage[data_name].dtypes
        
	def get_pdf_colstatistics(self,data_name:str):
		return self.storage[data_name].describe()
        
	def show_data(self):
		print(self.storage)

	def show_data_names(self):
		return list(self.storage.keys())
		
		
def check_data_compat(input_dict:dict, req_id:str):

	if req_id == 'df':

		# one dataframe input only
		if len(input_dict.get('df', [])) == 1 and len(input_dict.get('list', [])) == 0:
			return True
		else:
			print('[note] data source should contain only one dataframe')
			return False
		
	elif req_id == 'ddf':

		# two dataframe input only
		if len(input_dict.get('df', [])) == 2 and len(input_dict.get('list', [])) == 0:
			return True
		else:
			print('[note] data source should contain two dataframes')
			return False
		
	elif req_id == 'mdf':

		# multiple dataframe input only
		if len(input_dict.get('df', [])) > 1 and len(input_dict.get('list', [])) == 0:
			return True
		else:
			print('[note] data source should contain two or more dataframes')
			return False
		
	elif req_id == 'list':

		# one list input only
		if len(input_dict.get('list', [])) == 1 and len(input_dict.get('df', [])) == 0:
			return True
		else:
			print('[note] data source should contain only one list')
			return False
		
	elif req_id == 'dlist':

		# one list input only
		if len(input_dict.get('list', [])) == 2 and len(input_dict.get('df', [])) == 0:
			return True
		else:
			print('[note] data source should contain only one list')
			return False
		
	elif req_id == 'mlist':

		# multiple list input only
		if len(input_dict.get('list', [])) > 1 and len(input_dict.get('df', [])) == 0:
			return True
		else:
			print('[note] data source should contain only one list')
			return False
			
	elif req_id == 'alist':

		# 1+ lists 
		if len(input_dict.get('list', [])) >= 1 and len(input_dict.get('df', [])) == 0:
			return True
		else:
			print('[note] data source should contain only one list')
			return False
				
	elif req_id == 'list_df':

		# one list and one dataframe
		if len(input_dict.get('list', [])) == 1 and len(input_dict.get('df', [])) == 1:
			return True
		else:
			print('[note] data source should contain one list and dataframe')
			return False
	else:
		print('[note] no compatible data sources')
		return False
