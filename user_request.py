
import pandas as pd
from ner_parser import Parser
from tokenisers import punktokeniser, custpunkttokeniser
import difflib
import re


class user_request:

	'''
	########################################################################
	
	                           User Request Class
	
	########################################################################
	'''

	def __init__(self,data,modules):
		self.data = data  					# data class instance
		self.modules = modules				# module class instance
		
		self.token_info = dict()				# user request token information 	
		self.extracted_data = {}
		self.extracted_column = {}
		self.extracted_column_list = []
		self.extracted_params = {}

	def store_tokens(self,request:str):
		
		'''
		
		Tokenise User Request
		
		'''
		
		self.string = request
		self.tokens = custpunkttokeniser(request)
		self.add_column_token_info({'token':self.tokens})

	def evaluate(self):
		
		'''
		
		user request token length data extraction 
		
		'''
		
		# data related token info extraction
		self.data_in_tokens = [True if i in self.data.show_data_names() else False for i in self.tokens] 
		self.dtype_in_tokens = [self.data.dtype[i] if i in self.data.show_data_names() else None for i in self.tokens]
		self.add_column_token_info({'data_id':self.data_in_tokens})
		self.add_column_token_info({'dtype':self.dtype_in_tokens}) # data token type
	
		# add dataframe column
		self.check_tokens_for_pdf_columns() # self.column_in_tokens
		self.add_column_token_info({'col_id':self.column_in_tokens}) 
		
		# add [,] in between -column if it is missing
		# -column -column -> -column, -column
		self.adjust_column_series()

		# self.find_neighbouring_tokens()  # self.grouped_token_id
		self.column_name_groupings()     # self.grouped_column_idx/names
							         # grouped based on [and],[,] tokens 

		self.ac_tokens = [None for i in range(len(self.tokens))]
		self.token_info['ac_id'] = self.ac_tokens
		if(self.grouped_column_idx is not None):
			self.replace_tokens_to_columns() # use [grouped_column_idx] to adjust token_info

		# add module parameter tags
		self.preset_param_tagger()    # module preset parameter tagger
		self.add_column_token_info({'preset_param':self.param_preset_tags})
		
		# define token types (mainly for -value)
		self.set_token_type()
		self.add_column_token_info({'ttype':self.ttype_in_tokens}) # set token type
		
		# generalise tokens
		self.generalise_tokens()
		self.add_column_token_info({'mtoken':self.mtokens})
		
		'''
		
		Data Parameter and Column Extractions
		
		'''
		
		# extract and store the token data
		self.data_extraction()
		
		# extract PARAM parameters
		self.param_extraction() 
		# print(self.extracted_params)
		
		# extract general column references
		self.column_extraction() 	
		# print(self.extracted_column)
		# print(self.extracted_column_list)
		
		# print(self.show_token_info())
	
	def data_extraction(self):

		'''
		
		Extract Data Tokens
		
		'''
		
		data_token_idx = [ii for ii,i in enumerate(self.dtype_in_tokens) if i != None] # find all data tokens idx
		data_name = [self.tokens[idx] for idx in data_token_idx] # data name
		data_type = [self.dtype_in_tokens[idx] for idx in data_token_idx] # data type

		# acceptable data types
		data_storage = {'list':[],'df':[]}
		data_names = {'list':[],'df':[]}
		for name,type_id in zip(data_name,data_type):
			data_storage[type_id].append(self.data.storage[name])
			data_names[type_id].append(name)

		self.extracted_data['idx'] = data_token_idx  # data token index (list)
		self.extracted_data['name'] = data_name  # data name list (list)
		self.extracted_data['type'] = data_type  # data type list (list)
		self.extracted_data['storage'] = data_storage  # {'list':[],'df':[]} format the actual dataset 
		self.extracted_data['storage_name'] = data_names # {'list':[],'df':[]} format dataset names

		
	def column_extraction(self):
		
		'''
		
		Extract column only expressions; column references without parameters
		
		'''
		
		mtokens = self.mtokens # generalised tokens
		mtokens_string = ' '.join(self.mtokens) # up to date self.string

		mtoken_spans = {}	
		start_idx = 0
		for ii,token in enumerate(mtokens):
			start_idx = mtokens_string.find(token, start_idx)
			end_idx = start_idx + len(token) - 1  # Adjust end index to be inclusive
			mtoken_spans[(start_idx,end_idx)] = ii	
			start_idx = end_idx + 1

		token_spans = {}
		for key,value in mtoken_spans.items():
			token_name = self.tokens[value]
			gtoken_name = self.mtokens[value]
			ac_name = self.ac_tokens[value]
			
			if(gtoken_name.startswith('-')):
				if(gtoken_name == '-column'):
					token_spans[key] = token_name
				elif(gtoken_name == '-columns'):
					token_spans[key] = ac_name
			else:
				token_spans[key] = None
			
		# define patterns for matching
		pattern_af = "(for each|for every|each|every|of|for|for column|for columns|for the column|in the column|in|columns) -column"
		
		patterns = []
		pattern_after = pattern_af
		patterns.append(pattern_after)
		
		# pattern_matches : re matches for mtoken
		pattern_matches = {}
		for pattern in patterns:
			match = list(re.finditer(pattern, mtokens_string))
			for i in match:
				pattern_matches[i.span()] = i.group()
				
		if(len(pattern_matches) > 0):
			print('\n> Column references found \n')
			print(pattern_matches)
				
		# find tuple spans in dictionary of token spans
		def get_values_in_range(tuples_list, dictionary):
			result = []
			for span in tuples_list:
				start, end = span
				for key in dictionary:
					key_start, key_end = key
					if (key_start >= start and key_end <= end) or (key_start <= start and key_end >= end):
						result.append(dictionary[key])
						
			return result
			
		# for all regular expression matches keep only column references
		all_column_matches = []
		for span,token in pattern_matches.items():
			tuples_list = [span]
			token_matches = get_values_in_range(tuples_list,token_spans)	
			column_name_list = [i for i in token_matches if i != None]
			column_name_str = column_name_list[0]
			all_column_matches.append(column_name_str)
			self.extracted_column[token] = column_name_str
		
		self.extracted_column_list = all_column_matches
		
		if(len(pattern_matches)>0):
			print(self.extracted_column)
			print(self.extracted_column_list)
			
	
	@staticmethod
	def isfloat(strs:str):
		if(re.match(r'^-?\d+(?:\.\d+)$', strs) is None):
			return False
		else:
			return True
	
	@staticmethod		
	def isint(strs:str):
		if(re.match(r'\w+',strs) is None):
			return False
		else:
			return True
			
	def param_extraction(self):
		
		"""
		
		Extracted specified parameters using regular expression
		
		"""
		
		mtokens = self.mtokens # generalised tokens
		tokens = self.tokens # main tokens
		mtokens_string = ' '.join(self.mtokens) # up to date self.string

		mtoken_spans = {}	
		start_idx = 0
		for ii,token in enumerate(mtokens):
			start_idx = mtokens_string.find(token, start_idx)
			end_idx = start_idx + len(token) - 1  # Adjust end index to be inclusive
			mtoken_spans[(start_idx,end_idx)] = ii
			start_idx = end_idx + 1

		token_spans = {}
		for key,value in mtoken_spans.items():

			token_name = self.tokens[value]
			gtoken_name = self.mtokens[value]
			ac_name = self.ac_tokens[value]
				
			if(gtoken_name.startswith('~')):
				token_spans[key] = self.mtokens[value]
			elif(gtoken_name == '-columns'):
				token_spans[key] = self.ac_tokens[value]
			else:
				token_spans[key] = self.tokens[value]
			
		# define patterns for matching
		pattern_bef = r"(set to|equal to|as|stored as)"
		pattern_af = r"(as|equal|=|:|to|of)"
		
		patterns = []
		pattern_before = r'-\w+ ' + pattern_bef + r' ~\w+'
		pattern_after = r'~\w+ ' + pattern_af + r' -\w+'

		patterns.append(pattern_before)
		patterns.append(pattern_after)
			
		pattern_matches = {}
		for pattern in patterns:
			match = list(re.finditer(pattern, mtokens_string))
			for i in match:
				pattern_matches[i.span()] = i.group()
				
		if(len(pattern_matches) > 0):
			print('\n> Pattern references found \n')
			print(pattern_matches)
				
		# find tuple spans in dictionary of token spans
		def get_values_in_range(tuples_list, dictionary):
			result = []
			for span in tuples_list:
				start, end = span
				for key in dictionary:
					key_start, key_end = key
					if (key_start >= start and key_end <= end) or (key_start <= start and key_end >= end):
						result.append(dictionary[key])
			return result
		
		# if matches are found, extract parameters	
		if(len(pattern_matches)>0):
			
			# for all regular expression matches
			for span,token in pattern_matches.items():
				tuples_list = [span]
				results = get_values_in_range(tuples_list,token_spans)
				del results[1:-1] # keep only first and last token
				
				# put them in the predefined order
				if('~' in results[0]):
					pass
				elif('~' in results[-1]):
					results = sorted(results,reverse=True)
				
				param_name = results[0][1:]
				param_value = results[1]
				
				# check types
				float_check = self.isfloat(param_value)
				int_check = self.isint(param_value)
				
				if(float_check):
					param_value = float(param_value)
				elif(int_check):
					param_value = int(param_value)
					
				# store parameter
				self.extracted_params[param_name] = param_value
				
		if(len(pattern_matches)>0):
			print('extracted parameters')
			print(self.extracted_params)
	
	'''
	
	Parameter Token/Value Extraction
	
	label_params_names : label tag names with [~]
	label_params : replace string token names, floats, integers with [-] identifier
	
	'''

	@staticmethod
	def label_param_names(ls:pd.DataFrame):

		"""
		
		[ 1. add [~] labels to PARAM tokens ]
			
			B-PARAM, I-PARAM, M-PARAM
		
		eg. [~x] column [~y] ...
		
		"""

		ls = ls.copy()
		# indicies at which column data is available
		ner_param_idx = ls[ls['tag_id'].isin(['B-PARAM','I-PARAM'])].index.tolist() 
		ls.loc[ner_param_idx,'token'] = "~" + ls['token']
		return ls
		

	def preset_param_tagger(self):
		
		"""
		
		Tag module token parameters found in user request
		
		"""
		
		preset_params = self.modules.token_mparams
		tokens = self.tokens
		lst_param_tags = ['O']*len(tokens)
		for ii,i in enumerate(tokens):
			if(i in preset_params):
				lst_param_tags[ii] = 'B-PARAM'

		self.param_preset_tags = lst_param_tags

		
		
	def generalise_tokens(self):
		
		"""
		
		Generalise tokens so they can be used for classification
		
		"""

		temp = self.tokens.copy()
		for ii,i in enumerate(self.tokens):
			
			if(self.column_in_tokens[ii] != None):
				temp[ii] = '-column'
				
			if(self.dtype_in_tokens[ii] != None):
				temp[ii] = '-' + self.dtype_in_tokens[ii]
				
			if(self.ttype_in_tokens[ii] in ['float','int']):
				temp[ii] = '-value'
				
			if(self.param_preset_tags[ii] == 'B-PARAM'):
				temp[ii] = '~' + temp[ii]
				
		self.mtokens = temp
		
	def check_tokens_for_pdf_columns(self):

		"""
		
		Check if request token is in request data token column names
		
		"""
		
		self.column_in_tokens = [None] * len(self.tokens)

		# loop through data tokens
		for data_token,token in zip(self.data_in_tokens,self.tokens):

			if(data_token):
				try:
					data_columns = self.data.get_pdf_colnames(token)
					for ii,ltoken in enumerate(self.tokens):
						if(ltoken in data_columns):
							self.column_in_tokens[ii] = token
				except:
					pass


	def find_neighbouring_tokens(self):

		"""
		
		group together tokens which are connected by [and], [,]
		returning their indicies only
		
		"""

		# find and and , tokens
		comma_indices = [i for i, token in enumerate(self.tokens) if token == ',']
		and_indices = [i for i, token in enumerate(self.tokens) if token == 'and']
		merging_tokens = [',','and']

		# store their neighbours names
		tuples_list = []
		for ii,token in enumerate(self.tokens):
			if(token in merging_tokens):
				tuples_list.append([ii-1,ii+1])   

		# merge neighbouring groups
		if(len(tuples_list) > 1):
			merged_tuples = None
			for ii in range(len(tuples_list) - 1):
				tlist = tuples_list[ii].copy()
				tlist.extend(tuples_list[ii+1])
				merged_tuples = set(tlist)
				
			self.grouped_token_idx = list(merged_tuples)
			
		elif(len(tuples_list) == 1):
			self.grouped_token_idx = list(set(tuples_list[0]))
		else:
			self.grouped_token_idx = None

		
	def column_name_groupings(self):
	
		"""
		
		Group together pandas dataframe column names

		[-column] [x] [-column] -> [-columns]
		
		[grouped_column_idx] token indicies 
		[grouped_column_names] token names
		
		"""

		columns = self.column_in_tokens
		tokens = self.tokens
		
		# store indicies which have the same data source to the left and to the right
		grouped_indices = []
		for i in range(1, len(columns) - 1):
			if ((columns[i-1] == columns[i+1]) and columns[i-1] != None and columns[i+1] != None and tokens[i] in [',','and']):
				grouped_indices.append(i)
				
		if(len(grouped_indices) > 0):
		
			print('\n> Found active column grouping \n')
		
			lst_groups = []
			for group in grouped_indices:
				lst_groups.append([group-1,group,group+1])
	
			idx_to_token = {i:j for i,j in enumerate(self.tokens)}
	
			def merge_nested_lists(nested_lists):
				merged_lists = []
				current_list = nested_lists[0]
	
				for sublist in nested_lists[1:]:
					if current_list[-1] == sublist[0]:
						# Check for duplicates before extending
						overlap = [element for element in sublist if element not in current_list]
						current_list.extend(overlap)
					else:
						merged_lists.append(current_list)
						current_list = sublist
	
				merged_lists.append(current_list)
				return merged_lists
			
			self.grouped_column_idx = merge_nested_lists(lst_groups)
			self.grouped_column_names = [[idx_to_token.get(item) for item in sublist] for sublist in self.grouped_column_idx]
				
			print(self.grouped_column_names)
			
		else:
			self.grouped_column_idx = None
			self.grouped_column_names = None
			
		
	'''
	
	[Token Information Related]
	
	'''

	# add new column to token_info        
	def add_column_token_info(self,column:dict):
		self.token_info.update(column)
	
	# show user request token information	
	def show_token_info(self):
		print(pd.DataFrame(self.token_info))

	# return the dataframe format of token_info
	def get_token_info(self):
		return pd.DataFrame(self.token_info)
	
	# remove token_info indicies
	def remove_idx_token_info(self,idx_remove:list):
	
		idx_remove = sorted(idx_remove,reverse=True)	
		for key in self.token_info:
			for idx in idx_remove:
				self.token_info[key].pop(idx)

	# add values to token_info index 	
	def add_values_to_token_info(self, values_to_add:dict):
		for key, values in values_to_add.items():
			if key in self.token_info:	
				for index, value in values.items():
					if index < len(self.token_info[key]):
						self.token_info[key].insert(index, value)
					else:
						print(f"Index {index} is out of range for list '{key}'")

	# replace token values for token_info index 	
	def replace_values_to_token_info(self, values_to_add:dict):
		for key, values in values_to_add.items():
			if key in self.token_info:	
				for index, value in values.items():
					if index < len(self.token_info[key]):
						self.token_info[key][index] = value
					else:
						print(f"Index {index} is out of range for list '{key}'")
						
	# add [,] and [and] between -columns that aren't specified
	def adjust_column_series(self):
		for ii in range(len(self.column_in_tokens)-1):
			if((self.column_in_tokens[ii] == self.column_in_tokens[ii+1]) and self.column_in_tokens[ii] != None):
				self.add_values_to_token_info({
									 	 	 	'token':{ii+1:','},
												'data_id':{ii+1:False}, 
												'dtype':{ii+1:None},
												'col_id':{ii+1:None}
												})
				
	def replace_tokens_to_columns(self):

		"""
		
		Replace grouped multiple [-column] with [-columns]
		
		"""
		
		for group,names in zip(self.grouped_column_idx,self.grouped_column_names):
			
			# replace the first index of the group
			tgroup = group.copy()
			self.replace_values_to_token_info({
											'token':{group[0]:'-columns'},
											'data_id':{group[0]:False}, 
											'dtype':{group[0]:None},
											'col_id':{group[0]:None},
											'ac_id': {group[0]:[i for i in names if i not in [',','and']]}
											})
											
			tgroup.pop(0)

			# add -remove idx tags to the remaining indicies
			for idx in tgroup:

				modify = {
							'token':{idx:'-remove'},
							'data_id':{idx:False}, 
							'dtype':{idx:None},
							'col_id':{idx:None},
							'ac_id':{idx:None}
						}

				self.replace_values_to_token_info(modify)

		# find the indicies to be removed
		idx_remove = []
		for ii,token in enumerate(self.tokens):
			if(token == '-remove'):
				idx_remove.append(ii)

		# remove -remove tokens from token_info
		self.remove_idx_token_info(idx_remove)

	@staticmethod
	def string_diff_index(ref_string:str,string:str):

		"""
		
		Finds the difference in string inputs
		
		"""

		# Tokenize both strings
		reference_tokens = ref_string.split()
		second_tokens = string.split()

		# Find the indices of removed tokens
		removed_indices = []
		matcher = difflib.SequenceMatcher(None, reference_tokens, second_tokens)
		for op, i1, i2, _, _ in matcher.get_opcodes():
			if op == 'delete':
				removed_indices.extend(range(i1, i2))

		return removed_indices
	
	def set_token_type(self):
		
		"""
		
		Sets user request token type
		
		"""

		lst_types = []
		for token in self.tokens:

			if(self.isfloat(token)):
				type_id = 'float'
			elif(token.isnumeric()):
				type_id = 'int'
			else:
				type_id = 'str'

			lst_types.append(type_id)

		self.ttype_in_tokens = lst_types