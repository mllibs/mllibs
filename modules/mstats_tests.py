
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from nlpm import parse_json
from scipy import stats
from scipy.stats import kstest, shapiro, chisquare, jarque_bera, f_oneway
import json
import os

class stats_tests:
	
	def __init__(self):
		self.name = 'stats_tests'
		path = 'modules/mstats_tests.json'    
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
		self.data_format = args['pred_info']['data_compat'] # input data format
		self.columns = args['column_list']

		self.params = args['params']
		for key,value in self.params.items():
			args[key] = value

		if(self.select == 'its_ttest'):
			self.its_ttest(args)
		elif(self.select == 'd_ttest'):
			self.d_ttest(args)
		elif(self.select == 'os_ttest'):
			self.os_ttest(args)
		elif(self.select == 'u_test'):
			self.utest(args)
			
		elif(self.select == 'ksmirnov_tests'):
			
			if(args['sub_task'] == 'normal'):
				self.kstest_onesample_normal(args)
			elif(args['sub_task'] == 'uniform'):
				self.kstest_onesample_uniform(args)
			elif(args['sub_task'] == 'exponential'):
				self.kstest_onesample_exponential(args)

		elif(self.select == 'shapiro_wilk'):
			self.shapirowilk_normal(args)
		elif(self.select == 'one_anova'):
			self.oneway_anova(args)


		return self.result
	
	"""
	=========================================================

	Activation Functions
	
	=========================================================
	"""

	"""
	
	Student's t-test

	"""
			
	def its_ttest(self,args:dict):

		"""
		
		[Independent Student's t-test]

		"""

		if(self.data_format == 'dlist'):
			lsts = self.data['list']
		
		statistic, p_value = stats.ttest_ind(*lsts)
		
		print("t-statistic:", statistic)
		print("pvalue:", p_value)
		
		# Compare p-value with alpha
		if p_value < 0.05:
			print("Reject the null hypothesis : there is a significant difference between the two groups.")
		else:
			print("Fail to reject the null hypothesis : there is no significant difference between the two groups.")

		self.result = p_value


	def d_ttest(self,args:dict):

		"""
		
		[Dependent Student's t-test]

		"""

		if(self.data_format == 'dlist'):

			lsts = self.data['list']
			lsta,lstb = lsts[0],lsts[1]
		
			if(len(lsta) == len(lstb)):

				statistic, p_value = stats.ttest_rel(lsta,lstb)
				
				print("t-statistic:", statistic)
				print("pvalue:", p_value)
			
				# Compare p-value with alpha
				if p_value < 0.05:
					print("Reject the null hypothesis : there is a significant difference between the two groups.")
				else:
					print("Fail to reject the null hypothesis : there is no significant difference between the two groups.")

				self.result = p_value
			
			else:
				print(len(lsta),len(lstb))
				print('samples must have the same length')


	def os_ttest(self,args:dict):
		
		"""

		[one sample t-test]

		compare the mean of a single group to a known 
		population mean or a specific value.

		"""

		if(self.data_format == 'list'):
			lst = self.data['list'][0]
			
		# the population mean needs to be specified
		if(args['popmean'] != None):

			# Perform one-sample t-test
			statistic, p_value = stats.ttest_1samp(lst,popmean=args['popmean'])
			
			print("t-statistic:", statistic)
			print("pvalue:", p_value)
			
			# Check if the null hypothesis can be rejected
			if p_value < 0.05:
				print(f"Reject the null hypothesis: The mean is significantly different from {args['popmean']}")
			else:
				print(f"Fail to reject the null hypothesis: The mean is not significantly different from {args['popmean']}")

			self.result = p_value

		else:
			
			print('[note] please specify the population mean using popmean')

			
	def utest(self,args:dict):
		
		"""

		Perform Mann-Whitney U-test

		"""
		
		if(self.data_format == 'dlist'):
			lsts = self.data['list']

		statistic, p_value = stats.mannwhitneyu(*lsts)
		
		print("u statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value < 0.05:
			print("Reject null hypothesis : There is a significant difference between the two samples.")
		else:
			print("Fail to reject null hypothesis : There is no significant difference between the two samples.")
			
		self.result = p_value
		
		
	"""
	
	Kolmogorov Smirnov Tests

	"""
		
	def kstest_onesample_normal(self,args:dict):
		
		"""

		Perform Kolmogorov-Smirnov test for [normal] distribution

		"""
		
		if(self.data_format == 'list'):
			lst = self.data['list'][0]
		statistic, p_value = stats.kstest(lst, 'norm')
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
		alpha = 0.05
		if p_value < alpha:
			print("Reject the null hypothesis: Data does not follow a normal distribution.")
		else:
			print("Fail to reject the null hypothesis: Data follows a normal distribution.")
			
		self.result = p_value
		
	def kstest_onesample_uniform(self,args:dict):
		
		"""

		Perform Kolmogorov-Smirnov test for [uniform] distribution

		"""
		
		if(self.data_format == 'list'):
			lst = self.data['list'][0]
		statistic, p_value = stats.kstest(lst, 'uniform')
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
		alpha = 0.05
		if p_value < alpha:
			print("Reject the null hypothesis: Data does not follow a uniform distribution.")
		else:
			print("Fail to reject the null hypothesis: Data follows a uniform distribution.")
			
		self.result = p_value
			
	def kstest_onesample_exponential(self,args:dict):
		
		"""

		Perform Kolmogorov-Smirnov test for [exponential] distribution

		"""
		
		if(self.data_format == 'list'):
			lst = self.data['list'][0]
		statistic, p_value = stats.kstest(lst, 'expon')
		
		print("ks statistic:", statistic)
		print("pvalue:", p_value)
		
		alpha = 0.05
		if p_value < alpha:
			print("Reject the null hypothesis: Data does not follow an exponential distribution.")
		else:
			print("Fail to reject the null hypothesis: Data follows an exponential distribution.")	
			
		self.result = p_value

	def shapirowilk_normal(self,args:dict):
		
		"""

		Shapiro Wilk Test for normality

		"""
			
		if(self.data_format == 'list'):
			lst = self.data['list'][0]
		statistic, p_value = stats.shapiro(lst)
		
		# Print the test statistic and p-value
		print("test statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value > 0.05:
			print("Fail to reject the null hypothesis : The data is normally distributed")
		else:
			print("Reject the null hypothesis : The data is not normally distributed")

		self.result = p_value


	def oneway_anova(self,args:dict):
		
		"""

		[One way ANOVA test]

		It is used to determine if there are any statistically significant 
		differences between the (means) of two or more groups

		"""

		if(self.data_format == 'mlist'):
			lst = self.data['list']

		statistic, p_value = stats.f_oneway(*lst)
		
		# Print the results
		print("statistic:", statistic)
		print("pvalue:", p_value)
		
		if p_value < 0.05:
			print("Reject the null hypothesis: The means of the groups are not equal")
		else:
			print("Fail to reject the null hypothesis: The means of the groups are equal")

		self.result = p_value