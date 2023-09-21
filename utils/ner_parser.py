#from IPython.display import clear_output
import pandas as pd
import numpy as np
import re
import warnings;warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize

'''

NER Annotation Class

'''
# Tagging format [tag : word] as per A

class ner_annotator:
	
	def __init__(self,df:pd.DataFrame):
		self.df = df
		self.word2tag = {}
		self.LABEL_PATTERN = r"\[(.*?)\]"
		self.deactive_df = None
		self.active_df = None
		
		self.__initialise()
		
	def __initialise(self):
		
		'''
		
		[1] ANNOTATION COLUMN RELATED OPERATIONS
		
		'''
		
		# if annotaion column is all empty
		
		if('annotated' in self.df.columns):
			
			if(self.df['annot'].isna().sum() == self.df.shape[0]):
				self.df['annot'] = None
				
			# if annotation column is not empty
				
			elif(self.df['annot'].isna().sum() != self.df.shape[0]):
				
				# Store Tags
				for idx,row_data in self.df.iterrows():
					
					# if its already been annotated
					if(type(row_data['annot']) == str):
						matches = re.findall(self.LABEL_PATTERN, row_data['annot'] )
						for match in matches:
							if(' : ' in match):
								tag, phrase = match.split(" : ")
								self.word2tag[phrase] = tag
							
		# if annotation column is not present
							
		else:
			word2tag = {}
			self.df['annot'] = None    
			
		# active_df -> NaN are present
		# deactive_df -> already has annotations
			
		self.active_df = self.df[self.df['annot'].isna()]
		self.deactive_df = self.df[~self.df['annot'].isna()]
		
	'''
	
	REVIEW ANNOTATIONS
	
	'''
	# nullify rows which are not NaN, but don't have 
		
	def review_annotations(self):
		idx = list(self.deactive_df[~self.deactive_df["annot"].str.contains(self.LABEL_PATTERN)]['annot'].index)
		annot = list(self.deactive_df[~self.deactive_df["annot"].str.contains(self.LABEL_PATTERN)]['annot'].values)
		
		for i,j in zip(idx,annot):
			print(i,j)
			
	# drop annotations (from deactive_df)
			
	def drop_annotations(self,idx:list):
		remove_df = self.deactive_df.iloc[idx]
		remove_df['annot'] = None
		self.active_df = pd.concat([self.active_df,remove_df])
		self.deactive_df = self.deactive_df.drop(list(idx),axis=0)
		self.deactive_df.sort_index()
		print('dopped annotations saving >> annot.csv')
		pd.to_csv('annot.csv',pd.concat([self.active_df,self.deactive_df]))
		
	'''
	
	ANNOTATE ON ACTIVE ONLY
	
	'''
		
	def ner_annotate(self):
		
		'''
		
		Cycle through all rows
		
		'''
	
		for idx,row_data in self.active_df.iterrows():
			
			# left hand side text data [text,annotation]
			q = row_data['text'] # question
			t = q                    # annotated [question holder]
			
			'''
			
			Start Annotating
			
			'''
			# q,t are not modified unless entered q
			
			annotate_row = True
			while annotate_row is True:
				
				print('Current Annotations:')
				print(t,'\n')
				
				# user input command (isn't modified)
				user = input('tag (word-tag) format >> ')
				
				# [1] end of annotation (go to next row)
				if(user in ['quit','q']):
					
					'''
					
					Quit Annotating Current Row 
					
					'''
					
					# [1] store annotation in dataframe
					row_data['annot'] = t
					
					# [2] store all found tags in dictionary (word2tag)
					
					# Store Tags (list of [X] matches]
					matches = re.findall(self.LABEL_PATTERN, t)
					
					# filter out incorrect matches
					temp_matches = []
					for match in matches:
						if(' : ' in match):
							temp_matches.append(match)
							
					matches = temp_matches
					
					for match in matches:
						tag, phrase = match.split(" : ")
						
						# if it hasn't already been added
						if(tag not in self.word2tag):
							self.word2tag[phrase] = tag
						
						# clean up output
						#               clear_output(wait=True)
						
						# [2] stop annotation loop
						
					annotate_row = False
						
				elif(user in 'stop'):
					
					'''
					
					Stop Annotating 
					
					'''
					
					ldf = pd.concat([self.deactive_df,self.active_df],axis=0)
					ldf.to_csv('src/mllibs/corpus/ner_mp.csv',index=False)
					continue
					# return
				
				# [3] Reset current Row Tags
				
				elif(user in ['reset','r']):
					
					t=q 
					print('[note] annotations have been reset!')
					print(t,'\n')
					
#					user = input('tag (word-tag) format >> ')
					
					# [4] Show current 
					
				elif(user == 'show'):
					print(self.word2tag)
					
				elif(user == 'dict'):
					
					print(self.word2tag)
					
					# tags
					tokenised_t = word_tokenize(t)
					set_tokenised = set(tokenised_t)
					set_dict = set(self.word2tag.keys())
					
					intersections = set_dict.intersection(set_tokenised) 
					print(intersections)
					
					# list of input characters
					lst_t = list(t)
					
					# for all found key intersections
					for word in intersections:
						
						tag = self.word2tag[word]
						express = f'[{tag} : {word}]'
						
						# function to remove cases found as part of word
						def remove_wordmatches(inputs:str,match:str):
							
							indicies = [(m.start(),m.end()) for m in re.finditer(match,inputs)]
							
							indicies_filter = []
							# if character is to left and right a[found]b
							for st_idx,end_idx in indicies:
								if(inputs[st_idx-1] == " " and inputs[end_idx] == " "):
									indicies_filter.append((st_idx,end_idx))
									
							return indicies_filter
							
						# filter out cases w/ idx-1 -> string
						matches = remove_wordmatches(t,word)
						print('matches',matches)
						
						# go through all found cases
						for match in matches:
							
							lst_temp = lst_t # temp list
							
							# get matching idx in list
							match_idxs = list(range(match[0],match[1]+1))
							word_len = match[1] - match[0]
							
							remove_ids = match_idxs.copy()
							remove_ids.pop(0)
							del lst_temp[remove_ids[0]:remove_ids[-1]]
							
							# replace annotation @ first mached index
							lst_temp[match[0]] = express
					
						# join everything back (t is updated)
						t = "".join(lst_t)

				elif('-' in user):
					
					# parse input
					word,tag = user.split('-')
					
					if(word == ''):
						word = input('please add word >> ')
					if(tag == ''):
						tag = input('please add tag >> ')
						
					if(word in t):
						express = f'[{tag} : {word}]' 
						
					# function to remove cases found as part of word
					def remove_wordmatches(inputs:str,match:str):
						
						inputs = inputs + " "
						indicies = [(m.start(),m.end()) for m in re.finditer(match,inputs)]
						
						indicies_filter = []
                        
						# if character is to left and right a[found]b
						for st_idx,end_idx in indicies:
							if(inputs[st_idx-1] == " " and inputs[end_idx] == " "):
								indicies_filter.append((st_idx,end_idx))
								
						return indicies_filter
						
					# filter out cases w/ idx-1 -> string
					matches = remove_wordmatches(t,word)
					
					# change string to list
					lst_temp = list(t)

					# go through all found cases
					for match in matches:
						
						# get matching idx in list
						match_idxs = list(range(match[0],match[1]+1))
							
						remove_ids = match_idxs.copy()
						remove_ids.pop(0)
						del lst_temp[remove_ids[0]:remove_ids[-1]]
						
						# replace annotation @ first mached index
						lst_temp[match[0]] = express

						t = "".join(lst_temp)
						
#					else:
#						print('not found in sentence')
						
				else:
					print('[note] please use (word-tag format)!')
					
		# finished annotation
		ldf = pd.concat([self.deactive_df,self.active_df],axis=0)
		ldf.to_csv('src/mllibs/corpus/ner_mp.csv',index=False)
					
					
#df_annot = pd.read_csv('sentence_splitters.csv')   # read dataframe
df_annot = pd.read_csv('src/mllibs/corpus/ner_mp.csv')   # read dataframe

temp = ner_annotator(df_annot)  #   start annotating documents
#temp.drop_annotations([3,4])   # drop annotations 
temp.review_annotations()       # show annotated rows
#print(temp.word2tag)          
temp.ner_annotate()
