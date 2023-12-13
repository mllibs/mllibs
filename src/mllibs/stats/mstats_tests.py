
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
import pingouin as pg
'''

Statistical Testing Module

'''

# Compare samples defined in lists

class stats_tests(nlpi):
    
    def __init__(self):
        self.name = 'stats_tests'  

        path = pkg_resources.resource_filename('mllibs','/stats/mstats_tests.json')
        with open(path, 'r') as f:
            self.json_data = json.load(f)
            self.nlp_config = parse_json(self.json_data)
        
    # select activation function
    def sel(self,args:dict):
                
        self.args = args
        select = args['pred_task']
        self.data_name = args['data_name']
        self.subset = args['subset']
        print('subset',self.subset)

        # [t-tests]
        
        if(select == 'its_ttest'):
            self.its_ttest(args)
        if(select == 'p_ttest'):
            self.paired_ttest(args)
        if(select == 'os_ttest'):
            self.os_ttest(args)

        # [u-test] [anova]

        if(select == 'utest'):
            self.utest(args)
        if(select == 'two_sample_anova'):
            self.two_sample_anova(args)

        # [check] Kolmogorov Smirnov Tests

        if(select == 'ks_sample_normal'):
            self.kstest_onesample_normal(args)
        if(select == 'ks_sample_uniform'):
            self.kstest_onesample_uniform(args)
        if(select == 'ks_sample_exponential'):
            self.kstest_onesample_exponential(args)

        # [check] Normality distribution checks

        if(select == 'lilliefors_normal'):
            self.lilliefors_normal(args)
        if(select == 'shapirowilk_normal'):
            self.shapirowilk_normal(args)
        if(select == 'jarque_bera_norma'):
            self.jarquebera_normal(args)

        # [check] chi2 tests

        if(select == 'chi2_test'):
            self.chi2_test(args)
        if(select == 'chi2_peng'):
            self.chi2_test_peng(args)


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

    # Student's t-test: This test is used to compare the [[means]] of (two independent samples) 
    # It assumes that the data is (normally distributed) and that the (variances of the 
    # two groups are equal)

    def its_ttest(self,args:dict):

        print('[note] assumption : data is normally distributed + variances of two groups are equal')

        statistic, p_value = stats.ttest_ind(args['data'][0], args['data'][1])

        print("T-statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    # [paired t-test]

    # This test is used when you have paired or matched observations.
    # It is used to determine if there is a significant difference between 
    # the means of two related groups or conditions.

    def paired_ttest(self,args:dict):

        print('[note] perform a paired two-sample t-test is used to compare the means of (two related groups)!')

        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(args['data'][0], args['data'][1])

        print("T-statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    # [one sample t-test]

    # This test is used when you want to compare the mean of a single group to a known population mean or a specific value.

    def os_ttest(self,args:dict):

        if(args['popmean'] != None):

            # Perform one-sample t-test
            t_statistic, p_value = stats.ttest_1samp(args['data'], popmean=args['popmean'])

            print("t-statistic:", statistic)
            print("P-value:", p_value)

            # Compare p-value with alpha
            if p_value <= 0.05:
                print("Reject the null hypothesis")
            else:
                print("Fail to reject the null hypothesis")

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

        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(args['data'][0], args['data'][1])

        print("U-statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    # [GENERAL] Kolmogorov Smirnov Test Two Sample Test for distribution

    def kstest_twosample(self,args:dict):

        # Perform the KS test
        statistic, p_value = kstest(args['data'][0], args['data'][1])

        print('[KS] test two samples from sample distribution')
        print("KS statistic:", statistic)
        print("P-value:", p_value)

    # Perform Kolmogorov-Smirnov test for [normal] distribution

    def kstest_onesample_normal(self,args:dict):

        statistic, p_value = kstest(args['data'], 'norm')

        print('[KS] test sample from (uniform) distribution')
        print("KS statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    # Perform Kolmogorov-Smirnov test for [Uniform] distribution

    def kstest_onesample_uniform(self,args:dict):

        statistic, p_value = kstest(args['data'], 'uniform')

        print('[KS] test sample from (uniform) distribution')
        print("KS statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    # Perform Kolmogorov-Smirnov test for Exponential distribution

    def kstest_onesample_exponential(self,args:dict):

        statistic, p_value = kstest(args['data'], 'expon')

        print('[KS] test sample from (exponential) distribution')
        print("KS statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis")

    # Lilliefors Test to check if distribution is normal distribution

    def lilliefors_normal(self,args:dict):

        # Perform the Lilliefors test
        statistic, p_value = lilliefors(args['data'])

        print("Lilliefors test statistic:", statistic)
        print("Lilliefors p-value:", p_value)
            
        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis") 

    # Shapiro-Wilk Test to check if distribution is normal

    def shapirowilk_normal(self,args:dict):

        # Perform Shapiro-Wilk test
        statistic, p_value = shapiro(args['data'])

        # Print the test statistic and p-value
        print("Test Statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis") 

    # [Chi2 statistical test]

    # Calculate a one-way chi-square test.
    # The chi-square test in scipy is a statistical test used to determine 
    # if there is a significant association between two categorical variables.

    # chi-square statistic measures how much the observed frequencies deviate 
    # from the expected frequencies. A higher value indicates a greater discrepancy.

    def chi2_test(self,args:dict):

        # perform the chi-squared test
        statistic, p_value = chisquare(args['data'][0], f_exp=args['data'][1])

        print("Chi-squared statistic:", statistic)
        print("P-value:", p_value)

        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis") 

    # [Chi2 statistical test] (pingouin version)

    # Chi-squared independence tests between two categorical variables
    # pg.chi2_independence(df, x='A',y='B') format, so input lists are 
    # converted to dataframes

    def chi2_test_peng(self,args:dict):

        data = pd.DataFrame({'first':args['data'][0],
                             'second':args['data'][1]})
        e,o,st=pg.chi2_independence(data=data,x='first',y='second')
        display(st[['pval','test']].round(3))

    # [ Jarque-Bera test ]

    # The Jarque-Bera test is a statistical test used to determine whether 
    # a given dataset follows a normal distribution. It is based on the 
    # skewness and kurtosis of the data. 

    def jarquebera_normal(self,args:dict):

        # Perform the Jarque-Bera test
        statistic, p_value = stats.jarque_bera(args['data'])

        print('Statistic:", statistic')
        print("P-value:", p_value)

        # Compare p-value with alpha (0.05)
        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis") 

    # [ ANOVA test ]

    # scipy.stats.f_oneway() is a function from the scipy library in Python that 
    # performs a one-way ANOVA (Analysis of Variance) test. 
    # It is used to determine if there are any statistically significant 
    # differences between the (means) of two or more groups

    def two_sample_anova(self,args:dict):
    
        # Perform one-way ANOVA test
        statistic, p_value = stats.f_oneway(args['data'][0], args['data'][1])

        # Print the results
        print("Statistic:", statistic)
        print("p-value:", p_value)

        if p_value <= 0.05:
            print("Reject the null hypothesis")
        else:
            print("Fail to reject the null hypothesis") 
