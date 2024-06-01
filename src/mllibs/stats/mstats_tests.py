
from mllibs.nlpi import nlpi
import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings; warnings.filterwarnings('ignore')
from mllibs.nlpm import parse_json
import pkg_resources
import json
from scipy import stats
from scipy.stats import kstest, shapiro, chisquare, jarque_bera, f_oneway
from statsmodels.stats.diagnostic import lilliefors
from mllibs.module_helper import confim_dtype, get_ddata, get_mdata, get_sdata, get_dmdata, get_spdata
import textwrap

class stats_tests(nlpi):
	
	'''

	Statistical Testing Module

	'''
	
	def __init__(self):
		self.name = 'stats_tests'  
		path = pkg_resources.resource_filename('mllibs','/stats/mstats_tests.json')
		with open(path, 'r') as f:
			self.json_data = json.load(f)
			self.nlp_config = parse_json(self.json_data)
			
	def sel(self,args:dict):
		
		'''

		Relevant Activation function is selected 

		'''
		
		self.args = args
		select = args['pred_task']
		self.data_name = args['data_name']
		self.subset = args['subset']
		self.info = args['task_info']['description']

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
			
		# [t-tests]
			
		if(select == 'its_ttest'):
			
			args['data'] = get_data('list','ddata')
			if(args['data'] != None):
				self.its_ttest(args)
			else:
				print('[note] please use two sources of list data')
				
		if(select == 'p_ttest'):
			
			args['data'] = get_data('list','ddata')
			if(args['data'] != None):
				self.paired_ttest(args)
			else:
				print('[note] please use two sources of list data')
			
		if(select == 'os_ttest'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):
				self.os_ttest(args)
			
		# [u-test] [anova]
			
		if(select == 'utest'):

			args['data'] = get_data('list','ddata')
			if(args['data'] != None):
				self.utest(args)

		if(select == 'oneway_anova'):

			args['data'] = get_data('list','dmdata')
			if(args['data'] != None): 
				self.oneway_anova(args)
			
		# [check] Kolmogorov Smirnov Tests
			
		if(select == 'ks_sample_normal'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):
				self.kstest_onesample_normal(args)

		if(select == 'ks_sample_uniform'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):
				self.kstest_onesample_uniform(args)

		if(select == 'ks_sample_exponential'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):
				self.kstest_onesample_exponential(args)
			
			# [check] Normality distribution checks
			
		if(select == 'lilliefors_normal'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):
				self.lilliefors_normal(args)

		if(select == 'shapirowilk_normal'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):	
				self.shapirowilk_normal(args)

		if(select == 'jarque_bera_normal'):

			args['data'] = get_data('list','sdata')
			if(args['data'] != None):
				self.jarquebera_normal(args)
			
		# [check] chi2 tests
			
		if(select == 'chi2_test'):
			
			args['data'] = get_data('list','ddata')
			if(args['data'] != None):
				self.chi2_test(args)
			
			
	# for converting numeric text into int/float
	def convert_str(self,key:str):
		try:
			try:
				# if args[key] is a string
				val = eval(self.args[key])
			except:
				# else just a value
				val = self.args[key]
		except:
			val = None
		return val
	
	'''

	Activation Functions

	'''
	
	# [independent two sample t-test]
	
	# Student's t-test: This test is used to compare the [means] of (two independent samples) 
	# It assumes that the data is (normally distributed) and that the (variances of the 
	# two groups are equal)
	
	def its_ttest(self,args:dict):
		
		statistic, p_value = stats.ttest_ind(*args['data'])
		
		print("t-statistic:", statistic)
		print("pvalue:", p_value)
		
		# Compare p-value with alpha
		if p_value < 0.05:
			print("Reject the null hypothesis : there is a significant difference between the two groups.")
		else:
			print("Fail to reject the null hypothesis : there is no significant difference between the two groups.")
			
	# [paired t-test]
	
	# This test is used when you have paired or matched observations.
	# It is used to determine if there is a significant difference between 
	# the means of two related groups or conditions.
			
	def paired_ttest(self,args:dict):
		
		statistic, p_value = stats.ttest_ind(*args['data'])

		if(len(args['data'][0]) == len(args['data'][1])):

			# Perform paired t-test
			statistic, p_value = stats.ttest_rel(lst_data[0],lst_data[1])
			
			print("t-statistic:", statistic)
			print("pvalue:", p_value)
				
			if p_value < 0.05:
				print("Reject the null hypothesis : there is a significant difference between the two sets of related data.")
			else:
				print("Fail to reject the null hypothesis : there is no significant difference between the two sets of related data.")
					
		else:
			print('[note] Both data sources must have the same length')

			
	def os_ttest(self,args:dict):
		
		'''

		[one sample t-test]

		This test is used when you want to compare the mean of a single group to a known 
		population mean or a specific value.

		'''
			
		if(args['popmean'] != None):
			
			# Perform one-sample t-test
			statistic, p_value = stats.ttest_1samp(args['data'][0], popmean=args['popmean'])
			
			print("t-statistic:", statistic)
			print("pvalue:", p_value)
			
			# Check if the null hypothesis can be rejected
			if p_value < 0.05:
				print(f"Reject the null hypothesis: The mean is significantly different from {args['popmean']}")
			else:
				print(f"Fail to reject the null hypothesis: The mean is not significantly different from {args['popmean']}")
		else:
			
			print('[note] please specify the population mean using popmean')
			
			
	# determine if there is a significant difference between the distributions
	
	# A : [u-test]
	
	# The [Mann-Whitney test], also known as the [Wilcoxon rank-sum test], 
	# is a nonparametric statistical test used to determine whether there 
	# is a significant difference between the distributions of two independent samples. 
	# It is often used when the data does not meet the assumptions of parametric tests 
	# like the t-test.
			
	def utest(self,args:dict):
		
		'''

		Perform Mann-Whitney U test

		'''

		statistic, p_value = stats.mannwhitneyu(*args['data'])
		
		print("u statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value < 0.05:
			print("Reject null hypothesis : There is a significant difference between the two samples.")
		else:
			print("Fail to reject null hypothesis : There is no significant difference between the two samples.")
			
	# [GENERAL] Kolmogorov Smirnov Test Two Sample Test for distribution
			
	def kstest_twosample(self,args:dict):
		
		# Perform the KS test
		statistic, p_value = kstest(*args['data'])
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
	def kstest_onesample_normal(self,args:dict):
		
		"""

		Perform Kolmogorov-Smirnov test for [normal] distribution

		"""
		
		statistic, p_value = stats.kstest(args['data'][0], 'norm')
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
		alpha = 0.05
		if p_value < alpha:
			print("Reject the null hypothesis: Data does not follow a normal distribution.")
		else:
			print("Fail to reject the null hypothesis: Data follows a normal distribution.")
			
			
	def kstest_onesample_uniform(self,args:dict):
		
		"""

		Perform Kolmogorov-Smirnov test for [uniform] distribution

		"""
		
		statistic, p_value = stats.kstest(args['data'][0], 'uniform')
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
		alpha = 0.05
		if p_value < alpha:
			print("Reject the null hypothesis: Data does not follow a uniform distribution.")
		else:
			print("Fail to reject the null hypothesis: Data follows a uniform distribution.")
			
			
	def kstest_onesample_exponential(self,args:dict):
		
		"""

		Perform Kolmogorov-Smirnov test for [exponential] distribution

		"""
		
		statistic, p_value = stats.kstest(args['data'][0], 'expon')
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
		alpha = 0.05
		if p_value < alpha:
			print("Reject the null hypothesis: Data does not follow an exponential distribution.")
		else:
			print("Fail to reject the null hypothesis: Data follows an exponential distribution.")
			
	def lilliefors_normal(self,args:dict):
		
		'''

		Lilliefors test for normality

		'''

		statistic, p_value = lilliefors(args['data'][0])
		
		print("Lilliefors statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value > 0.05:
			print("Fail to reject the null hypothesis : The data is likely to be normally distributed.")
		else:
			print("Reject the null hypothesis : The data is not likely to be normally distributed.")
			
			
	def shapirowilk_normal(self,args:dict):
		
		'''

		Shapiro Wilk Test for normality

		'''
			
		statistic, p_value = stats.shapiro(args['data'][0])
		
		# Print the test statistic and p-value
		print("test statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value > 0.05:
			print("Fail to reject the null hypothesis : The data is normally distributed")
		else:
			print("Reject the null hypothesis : The data is not normally distributed")
			
			
	def chi2_test(self,args:dict):
		
		'''
			
		# chi-square statistic measures how much the observed frequencies deviate 
		# from the expected frequencies. A higher value indicates a greater discrepancy.
		
		# perform the chi-squared test
		statistic, p_value = stats.chisquare(args['data'][0], f_exp=args['data'][1])
		
		'''

		print("chi2 statistic:", statistic)
		print("pvalue:", p_value)
		
		# Compare p-value with alpha (0.05)
		if p_value <= 0.05:
			print("Reject the null hypothesis : The observed frequencies are significantly different from the expected frequencies.")
		else:
			print("Fail to reject the null hypothesis : The observed frequencies are not significantly different from the expected frequencies.") 
			
			
	def jarquebera_normal(self,args:dict):
		
		# [ Jarque-Bera test ]
		
		# The Jarque-Bera test is a statistical test used to determine whether 
		# a given dataset follows a normal distribution. It is based on the 
		# skewness and kurtosis of the data. 
			
		# Perform the Jarque-Bera test
		statistic, p_value = stats.jarque_bera(args['data'][0])
		
		print("statistic:", statistic)
		print("pvalue:", p_value)
		
		# Compare p-value with alpha (0.05)
		if p_value <= 0.05:
			print("Reject the null hypothesis : The data does not follow a normal distribution")
		else:
			print("Fail to reject the null hypothesis : The data follows a normal distribution") 
			
			
	def oneway_anova(self,args:dict):
		
		'''

		[One way ANOVA test]

		# It is used to determine if there are any statistically significant 
		# differences between the (means) of two or more groups

		'''

		statistic, p_value = stats.f_oneway(*args['data'])
		
		# Print the results
		print("statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value < 0.05:
			print("Reject the null hypothesis: The means of the groups are not equal")
		else:
			print("Fail to reject the null hypothesis: The means of the groups are equal")
			