import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import json
import os
import pkg_resources
import matplotlib.pyplot as plt
import seaborn as sns


class stats_plot:
	
	def __init__(self):
		self.name = 'stats_plot'
		path = pkg_resources.resource_filename('mllibs','/modules/mstats_plot.json') 
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
		self.data_names = args['data']['storage_name'] # stored data names
		self.data_format = args['pred_info']['data_compat'] # input data forat
		self.columns = args['column_list']
		
		# store preextracted parameters
		self.params = args['params']
		for key,value in self.params.items():
			args[key] = value
			
		if(self.select == 'plot_hist'):
			self.dp_hist(args)
		elif(self.select == 'plot_kde'):
			self.dp_kde(args)
		elif(self.select == 'plot_box'):
			self.dp_box(args)

	"""
	=========================================================

	Activation Functions
	
	=========================================================
	"""
			
	def dp_hist(self,args:dict):

		"""

		Plot the Histrogram Univariate Distribution

		"""
		
		# input requires any number of lists
		if(self.data_format == 'alist'):
			lsts = self.data['list']

		# combine all lists into one dataframe
		lst_ldata = []
		ldata = self.data['list']
		lnames = self.data_names['list']
		for name,data in zip(lnames,ldata):
			ldata = pd.DataFrame(data,columns=['data'])
			ldata['sample'] = name
			lst_ldata.append(ldata)

		combined = pd.concat(lst_ldata)
		combined = combined.reset_index(drop=True)

		if(args['nbins'] is not None):
			bins = args['nbins']
		else:
			bins = 100 

		# plot the histograms

		fig, ax = plt.subplots(figsize=(10,4))
		sns.despine(left=True,right=True,top=True,bottom=True)
		plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)
		
		sns.histplot(combined,x='data',hue='sample',bins=bins,
					 alpha=0.5,edgecolor='k',linewidth=1,
					 ax=ax)

		ax.set_xlabel('Value')
		ax.set_ylabel('Frequency')
		ax.set_title('Univariate Distribution')
		plt.show()


	def dp_kde(self,args:dict):

		'''

		Plot the Kernel Density Univariate Distribution

		'''

        # [when lists] combine all dictionaries into one dataframe
		
		# input requires any number of lists
		if(self.data_format == 'alist'):
			lsts = self.data['list']

		# combine all lists into one dataframe
		lst_ldata = []
		ldata = self.data['list']
		lnames = self.data_names['list']
		for name,data in zip(lnames,ldata):
			ldata = pd.DataFrame(data,columns=['data'])
			ldata['sample'] = name
			lst_ldata.append(ldata)

		combined = pd.concat(lst_ldata)
		combined = combined.reset_index(drop=True)

        # plot the kernel density plot

		fig, ax = plt.subplots(figsize=(10,4))
		sns.despine(left=True,right=True,top=True,bottom=True)
		plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)

		# Create a kernel density plot
		sns.kdeplot(combined,x='data',hue='sample',
					ax=ax,fill=True)
		ax.set_xlabel('Value')
		ax.set_ylabel('Density')
		ax.set_title('Univariate Distribution')
		plt.show()


	def dp_box(self,args:dict):

		'''

		Plot the Boxplot Univariate Distribution

		'''

		# input requires any number of lists
		if(self.data_format == 'alist'):
			lsts = self.data['list']

		# combine all lists into one dataframe
		lst_ldata = []
		ldata = self.data['list']
		lnames = self.data_names['list']
		for name,data in zip(lnames,ldata):
			ldata = pd.DataFrame(data,columns=['data'])
			ldata['sample'] = name
			lst_ldata.append(ldata)

		combined = pd.concat(lst_ldata)
		combined = combined.reset_index(drop=True)

		# plot boxplot using seaborn

		fig, ax = plt.subplots(figsize=(10,4))
		sns.despine(left=True,right=True,top=True,bottom=True)
		plt.grid(linestyle='--', linewidth=0.5,alpha=0.2)
		
		sns.boxplot(combined,x='sample',y='data',width=0.5,
					ax=ax)

		ax.set_xlabel('Subset')
		ax.set_ylabel('Value')
		ax.set_title('Univariate Distribution')
		plt.show()
