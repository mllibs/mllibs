
from mllibs.nlpi import nlpi
import pandas as pd


'''

Check Data Specified Data Type Compatilibity with that of JSON

'''

def check_type_compatibility(ldata,tpred,input_format):

	if(ldata is not None):

		if(len(ldata) != len(input_format)):
			return False
		else:
			compat_format = []
			for iii,ii in enumerate(ldata):
				compat_format.append(isinstance(nlpi.data[ii]['data'],eval(input_format[iii])))
				if(all(compat_format) == True):
					return True
			else:
				return False
	else:
		return False

'''

Check Data Specified Data Type Compatilibity with that of JSON

'''

def confim_dtype(req_id:str,inputs:dict):

	'''

	Function used to confirm if activation function
	requirement [req_id] meets [inputs] (ldata dict) format

	'''

	def check_value_types(input_dict):

		lst_types = []
		for value in input_dict.values():
			if isinstance(value, list):
				lst_types.append('list')
			elif(isinstance(value,pd.DataFrame)):
				lst_types.append('pd.DataFrame')
			elif(isinstance(value,pd.Series)):
				lst_types.append('pd.Series')
			else:
				lst_types.append('None')

		return lst_types
			
	dtypes = (check_value_types(inputs))

	allocate_dtype = None
	if( len(set(dtypes)) == 1 and set(dtypes) == {'list'} and len(dtypes) > 2):
		allocate_dtype = 'mlist'
	elif( len(set(dtypes)) == 1 and set(dtypes) == {'list'} and len(dtypes) == 2):
		allocate_dtype = 'dlist'
	elif( len(set(dtypes)) == 1 and set(dtypes) == {'list'} and len(dtypes) == 1):
		allocate_dtype = 'slist'

	elif( len(set(dtypes)) == 1 and set(dtypes) == {'pd.DataFrame'} and len(dtypes) > 2):
		allocate_dtype = 'mdf'
	elif( len(set(dtypes)) == 1 and set(dtypes) == {'pd.DataFrame'} and len(dtypes) == 2):
		allocate_dtype = 'ddf'
	elif( len(set(dtypes)) == 1 and set(dtypes) == {'pd.DataFrame'} and len(dtypes) == 1):
		allocate_dtype = 'sdf'
	else:
		allocate_dtype = None

	if(allocate_dtype == req_id[0]): # if requirement is met
		return True
	elif(allocate_dtype == 'dlist' and req_id[0] == 'mlist'): # multi-list but requirement is double-list
		return True
	elif(allocate_dtype == 'ddf' and req_id[0] == 'mdf'):
		return True
	else:
		return False


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

dictionary {'list': ['asf'], 'df': []}
req df/list

'''

def get_single_element_value(dictionary:dict,req_id:str):

	single_element_values = []; single_element_key = []
	multi_element_values = []; multi_element_key = []
	for key, value in dictionary.items():
		if len(value) == 1:
			single_element_key.append(key)
			single_element_values.append(value[0])
		if len(value) > 1:
			multi_element_key.append(key)
			# multi_element_values.append(value[0])

	if(len(multi_element_key) > 1):
		print('[note] key with multiple entries found')
		return None
	else:
		if(len(single_element_key) > 1):
			print('[note] multiple keys with single entry found')
			return None
		else:
			if(single_element_key[0] == req_id):
				return single_element_values[0]


'''

	Functions that select particular subset of args['data']

'''
			
# get double data (single type req_id)
def get_ddata(dicts:dict,req_id:str):
	if(len(dicts[req_id]) == 2):
		return dicts[req_id] 
	else:
		return None
	
# get multiple data (single type req_id)
def get_mdata(dicts:dict,req_id:str):
	if(len(dicts[req_id]) > 2):
		return dicts[req_id]
	else:
		return None

# get double/multiple data (single type req_id)
def get_dmdata(dicts:dict,req_id:str):
	if(len(dicts[req_id]) >= 2):
		return dicts[req_id]
	else:
		return None

# get only one data (single type req_id)
def get_sdata(dicts:dict,req_id:str):
	if(len(dicts[req_id]) == 1):
		return dicts[req_id] 
	else:
		return None

# get one more more data (single type req_id)
def get_spdata(dicts:dict,req_id:str):
	if(len(dicts[req_id]) >= 1):
		return dicts[req_id] 
	else:
		return None


def get_nested_list_and_indices(input_list:list):

	'''

	input
	['island', ['bill_length_mm', 'bill_depth_mm']]

	output/return
	Index of nested list: 1
	Indices of rest of the elements: [0, 2]


	'''

	nested_count = 0; str_count = 0
	for i, item in enumerate(input_list):
		if isinstance(item, list) and len(item) == 2:
			nested_count += 1
		elif isinstance(item, str):
			str_count += 1

	if (nested_count > 1):
		print('[note] too much information has been given')

	elif(nested_count == 1):

		for i, item in enumerate(input_list):
			if isinstance(item, list) and len(item) == 2:
				rest_indices = [j for j in range(len(input_list)) if j != i]
				return i, rest_indices		
			elif isinstance(item, list):
				nested_index, rest_indices = get_nested_list_and_indices(item)
				if nested_index is not None:
					return nested_index, rest_indices

	elif(nested_count == 0 and str_count > 0):

		return None,list(range(len(input_list)))
		
	else:

		return None, None