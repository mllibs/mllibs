from mllibs.dict_helper import sfp, sgp, sfpne, column_to_subset
from mllibs.module_helper import confim_dtype, get_ddata, get_mdata, get_sdata, get_dmdata, get_spdata,get_nested_list_and_indices
from mllibs.nlpi import nlpi
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
from mllibs.df_helper import split_types
import pkg_resources
import json
import textwrap


# Define Palette
def hex_to_rgb(h):
	h = h.lstrip('#')
	return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

palette = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252']
palette_rgb = [hex_to_rgb(x) for x in palette]

'''

Standard seaborn library visualisations

'''

class eda_splot(nlpi):
	
	def __init__(self):
		self.name = 'eda_splot'  

		path = pkg_resources.resource_filename('mllibs', '/eda/meda_splot.json')
		with open(path, 'r') as f:
			self.json_data = json.load(f)
			self.nlp_config = parse_json(self.json_data)
			
		#default_colors_p = ['#b4d2b1', '#568f8b', '#1d4a60', '#cd7e59', '#ddb247', '#d15252'] # my custom (plotly)
		pallete = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
		self.default_colors = pallete
		
	# common functions
	  
	def set_palette(self,args:dict):
	  
		if(args['hue'] is not None):
			hueloc = args['data'][args['hue']]
			if(type(nlpi.pp['stheme']) is str):
				palette = nlpi.pp['stheme']
			else:
				palette = self.default_colors[:len(hueloc.value_counts())]
			
		else:
			hueloc = None
			palette = self.default_colors
		
		return palette

	def seaborn_setstyle(self):
		sns.set_style("whitegrid", {
		"ytick.major.size": 0.1,
		"ytick.minor.size": 0.05,
		"grid.linestyle": '--'
		})

	def sel(self,args:dict):
				
		'''
		
				Start of Activation Function Selection 

				input : args (module_args)
		
		'''

		# set l,global parameters
		select = args['pred_task']
		self.data_name = args['data_name']
		self.info = args['task_info']['description']
		sub_task = args['sub_task']
		column = args['column']

		'''

			Filter [module_args] to include only input parameters
		
		'''

		# remove everything but parameters
		keys_to_remove = ["task_info", "request",'pred_task','data_name','sub_task','dtype_req']
		args = {key: value for key, value in args.items() if key not in keys_to_remove}

		# update module_args (keep only non None)
		filtered_module_args = {key: value for key, value in args.items() if value is not None}
		args = filtered_module_args

		# get data 
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
		
		if(nlpi.silent == False):
			print('\n[note] module function info');print(textwrap.fill(self.info, 60));print('')

		''' 
		
		Activatation Function
		
		'''

		###################################################################################
		if(select == 'sscatterplot'):

			# get single dataframe
			args['data'] = get_data('df','sdata')
			if(args['data'] is not None):

				# columns w/o parameter treatment
				if(column != None):
					group_col_idx,indiv_col_idx = get_nested_list_and_indices(column)

					# group column names
					group_col = column[group_col_idx]

					# non grouped column names
					lst_indiv = []
					for idx in indiv_col_idx:
						lst_indiv.append(column[idx])

				'''

				Subset treatment options

				'''

				# [-column] and [-column]
				if(sub_task == 'xy_column'):
					try:
						args['x'] = group_col[0]
						args['y'] = group_col[1]
					except:
						pass
				elif(sub_task == 'param_defined'):
					pass

				# column is not needed anymore
				keys_to_remove = ["column"]
				args = {key: value for key, value in args.items() if key not in keys_to_remove}

				self.sscatterplot(args)

		###################################################################################
		elif(select == 'srelplot'):

			# get single dataframe
			args['data'] = get_data('df','sdata')
			if(args['data'] is not None):

				# columns w/o parameter treatment
				if(column != None):
					
					group_col_idx,indiv_col_idx = get_nested_list_and_indices(column)

					# group column names (if they exist)
					try:
						group_col = column[group_col_idx]
					except:
						pass

					# non grouped column names
					lst_indiv = []
					for idx in indiv_col_idx:
						lst_indiv.append(column[idx])

				'''

				Subset treatment options

				'''

				# [-column] and [-column]
				if(sub_task == 'xy_column'):
					try:
						args['x'] = group_col[0]
						args['y'] = group_col[1]
					except:
						pass
				  
				# [-column] and [-column] for all [-column]
				elif(sub_task == 'xy_col_column'):  
					try:
						args['x'] = group_col[0]
						args['y'] = group_col[1]
						args['col'] = lst_indiv[0]
					except:
						pass

				# [-column] and [-column] for all [-column] and for all [-column]
				elif(sub_task == 'xy_col_row'):  
					try:
						args['x'] = group_col[0]
						args['y'] = group_col[1]
						args['col'] = lst_indiv[0]
						args['row'] = lst_indiv[1]
					except:
						pass

				# parameters defined only [~x,~y]
				elif(sub_task == 'param_defined'):
					pass

				# parameters defined [~x,~y] for all [-column]
				elif(sub_task == 'param_defined_col'):
					args['col'] = lst_indiv[0]

				# parameters defined [~x,~y] for all [-column] and for all [-column]
				elif(sub_task == 'param_defined_col_row'):
					args['col'] = lst_indiv[0]
					args['row'] = lst_indiv[1]

				# column is not needed anymore
				keys_to_remove = ["column"]
				args = {key: value for key, value in args.items() if key not in keys_to_remove}

				# call relplot
				self.srelplot(args)

			else:
				print('[note] no dataframe data sources specified')

		###################################################################################
		elif(select == 'sboxplot'):
			self.sboxplot(args)
		elif(select == 'sresidplot'):
			self.sresidplot(args)
		elif(select == 'sviolinplot'):
			self.sviolinplot(args)
		elif(select == 'shistplot'):
			self.shistplot(args)
		elif(select == 'skdeplot'):
			self.skdeplot(args)
		elif(select == 'slmplot'):
			self.slmplot(args)
		elif(select == 'spairplot'):
			self.spairplot(args)
		elif(select == 'slineplot'):
			self.slineplot(args)
		elif(select == 'sheatmap'):
			self.sheatmap(args)
	
	'''
	
	Seaborn Scatter Plot [scatterplot]
	  
	'''
	  
	def sscatterplot(self,args:dict):
		  
		self.seaborn_setstyle()
		if('hue' in args):
			palette = self.set_palette(args)
			args['palette'] = palette
		if('mew' in args):
			args['linewidth'] = args['mew']
			del args['mew']
		if('mec' in args):
			args['edgecolor'] = args['mec']
			del args['mec']
		if(nlpi.pp['figsize']):
			figsize = nlpi.pp['figsize']
		else:
			figsize = None
		  
		plt.figure(figsize=figsize)
		sns.scatterplot(**args)
		
		sns.despine(left=True,bottom=True,right=True,top=True)
		if(nlpi.pp['title']):
			plt.title(nlpi.pp['title'])
			plt.tight_layout()
		plt.show()
		nlpi.resetpp()
		
	'''
	
	Seaborn scatter plot with Linear Model [lmplot]
	  
	'''
		
	def slmplot(self,args:dict):
	
		self.seaborn_setstyle()
		
		sns.lmplot(x=args['x'], 
				   y=args['y'],
				   hue=args['hue'],
				   col=args['col'],
				   row=args['row'],
				   data=args['data']
				  )
		
		sns.despine(left=True,bottom=True,right=True,top=True)
		if(nlpi.pp['title']):
			plt.subplots_adjust(top=0.90)
			g.fig.suptitle(nlpi.pp['title'])
			plt.tight_layout()
		plt.show()
		
	'''
	
	Seaborn Relation Plot

	
	'''

	def srelplot(self,args:dict):
			
		self.seaborn_setstyle()
		if('hue' in args):
			palette = self.set_palette(args)
			args['palette'] = palette
		if('mew' in args):
			args['linewidth'] = args['mew']
			del args['mew']
		if('mec' in args):
			args['edgecolor'] = args['mec']
			del args['mec']
		if(nlpi.pp['figsize']):
			args['height'] = nlpi.pp['figsize'][0]

		g = sns.relplot(**args)
		
		sns.despine(left=True,bottom=True,right=True,top=True)

		if(nlpi.pp['title']):
			plt.subplots_adjust(top=0.90)
			g.fig.suptitle(nlpi.pp['title'])
			plt.tight_layout()

		plt.show()
		nlpi.resetpp()
		
	'''
	
	Seaborn Box Plot [sns.boxplot]
	  
	'''
		
	def sboxplot(self,args:dict):
		
		palette = self.set_palette(args)
		self.seaborn_setstyle()
		
		if(args['bw'] is None):
			bw = 0.8
		else:
			bw = eval(args['bw'])
		
		sns.boxplot(x=args['x'], 
					y=args['y'],
					hue=args['hue'],
					width=bw,
					palette=palette,
					data=args['data'])
		
		sns.despine(left=True, bottom=True)
		if(nlpi.pp['title']):
			plt.title(nlpi.pp['title'])
		plt.show()
		
	'''
	
	Seaborn Violin Plot [sns.violinplot]
	  
	'''
		
	def sviolinplot(self,args:dict):
		
		palette = self.set_palette(args)
		self.seaborn_setstyle()
			
		sns.violinplot(x=args['x'], 
					   y=args['y'],
					   hue=args['hue'],
					   palette=palette,
					   data=args['data'],
					   inner="quart",
					   split=True
					   )   
		
		sns.despine(left=True, bottom=True)
		if(nlpi.pp['title']):
			plt.title(nlpi.pp['title'])
		plt.show()
		nlpi.resetpp()
		
	@staticmethod
	def sresidplot(args:dict):
	  
		sns.residplot(x=args['x'], 
					  y=args['y'],
					  color=nlpi.pp['stheme'][1],
					  data=args['data'])
		
		sns.despine(left=True, bottom=True)
		plt.show()
		
	'''
	
	Seaborn Histogram Plot [sns.histplot]
	  
	'''
	  
	def shistplot(self,args:dict):
		
		self.seaborn_setstyle()
	
		# default parameters (pre) & allowable parameters (allow)
		pre = {'nbins':'auto','barmode':'stack'}
		allow = {'barmode':['layer','dodge','stack','fill']}
		
		# set default parameter if not set
		nbins = sfp(args,pre,'nbins')
		barmode = sfp(args,pre,'barmode')
		palette = self.set_palette(args)
		
		# check if string is in allowable parameter
		if(barmode not in allow['barmode']):
			barmode = allow['barmode'][0]
			print('[note] allowable barmodes: [layer],[dodge],[stack],[fill]')
		  
		if(args['x'] is None and args['y'] is None and column is not None):
			args['x'] = column
			print('[note] please specify orientation [x][y]')
		
		sns.histplot(
					  x=args['x'], 
					  y=args['y'],
					  hue=args['hue'],
					  alpha = args['alpha'],
					  linewidth=args['mew'],
					  edgecolor=args['mec'],
					  data=args['data'],
					  palette=palette,
					  bins=nbins,
					  multiple=barmode
		)
		
		sns.despine(left=True, bottom=True)
		if(nlpi.pp['title']): 
			plt.title(nlpi.pp['title'])
		plt.show()
		nlpi.resetpp()
		
	'''
	
	Seaborn Kernel Density Plot
	
	'''

	def skdeplot(self,args:dict):
		  
		palette = self.set_palette(args)
			
		self.seaborn_setstyle()
		
		sns.kdeplot(x=args['x'],
					y=args['y'],
					hue = args['hue'],
					palette=palette,
					fill=nlpi.pp['fill'],
					data = args['data'])
		
		sns.despine(left=True, bottom=True)
		if(nlpi.pp['title']):
			plt.title(nlpi.pp['title'])
		plt.show()
		nlpi.resetpp()
		
	def seaborn_pairplot(self,args:dict):
   
		num,cat = split_types(args['data'])
			
		if(args['hue'] is not None):
			hueloc = args['hue']
			num = pd.concat([num,args['data'][args['hue']]],axis=1) 
			subgroups = len(args['data'][args['hue']].value_counts())
			if(type(nlpi.pp['stheme']) is list):
				palette = nlpi.pp['stheme'][:subgroups]
			else:
				palette = nlpi.pp['stheme']
		else:
			hueloc = None
			palette = nlpi.pp['stheme']
		
			
		sns.set_style("whitegrid", {
			"ytick.major.size": 0.1,
			"ytick.minor.size": 0.05,
			'grid.linestyle': '--'
		 })
			 
		sns.pairplot(num,
					 hue=hueloc,
					 corner=True,
					 palette=palette,
					 diag_kws={'linewidth':nlpi.pp['mew'],
							   'fill':args['fill']},
					 plot_kws={'edgecolor':args['mec'],
							   'linewidth':args['mew'],
							   'alpha':args['alpha'],
							   's':args['s']})   
		
		sns.despine(left=True, bottom=True)
		plt.show()
		nlpi.resetpp()
		
	'''
	
	Seaborn Line Plot 
	
	'''

	def slineplot(self,args:dict):
	
		self.seaborn_setstyle()
		palette = self.set_palette(args)

		sns.lineplot(x=args['x'], 
					 y=args['y'],
					 hue=args['hue'],
					 alpha=args['alpha'],
					 linewidth=args['mew'],
					 data=args['data'],
					 palette=palette)
		
		sns.despine(left=True, bottom=True)
		if(nlpi.pp['title']):
			plt.title(nlpi.pp['title'])
		plt.show()
		nlpi.resetpp()

	# seaborn heatmap
				
	def sheatmap(self,args:dict):
		
		if(args['hue'] is not None):
			hueloc = args['data'][args['hue']]
			if(type(nlpi.pp['stheme']) is str):
				palette = nlpi.pp['stheme']
			else:
				palette = palette_rgb[:len(hueloc.value_counts())]
				
		else:
			hueloc = None
			palette = palette_rgb
		
		num,_ = self.split_types(args['data'])
		sns.heatmap(num,cmap=palette,
					square=False,lw=2,
					annot=True,cbar=True)    
					
		plt.show()
		nlpi.resetpp()
	