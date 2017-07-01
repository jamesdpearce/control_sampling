import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def assign_control(x, control_group_name = ' ', control_size = 0.5):
    """
        Computes the wether or not x is in the control group.
    """
    random.seed(str(x)+control_group_name)
    return random.random() <= control_size

def make_control(df, id_col, control_cols = [], control_group_name = '_', 
                 control_size = 0.5, alpha = 0.05, control_label = 'control_group'): 
    
    """ 
        This function divides a dataframe up into control and treatment (not control) groups based on row id. 
        Control/treatment groups are formed by taking a hash of the concatination of the row id and the 
        control_group_name. This hash is them used as a seed to the python random rumber generator, the output
        of which is used to assign to a control/treatment group. 
        
        This function can take an arbitrary number of control columns, which are used in either a KS or Chi^2 
        test against the null hypothesis (that the distributions are the same). In other words, we expect these 
        statistical test to fail if the control group is chosen randomly, i.e. the p-values should be >> 0.05. 
        If statistically significant differences are found then the function should be rerun with a 
        different control_group_name. 
        
        
        Args
        -----
        
        df: Pandas dataframe object. This should include the Id columns id_col as well as all control columns. 
        
        id_col: Column that one wishes to use to determine the control and treatment groups. 
        
        control_cols: A list of columns to be checked after the control group has been made. 
        
        control_group_name: A string to label the particular control/treatment group set. This should be changed 
        for every experiment. 
        
        control_size: ratio of sample that should go into the control group
        
        alpha: 1 - confidence level, 0.05 means there is a 1/20 chance that the distributions will pass the 
        statistical test by chance alone. 
        
        Returns
        --------
        
        1) df is augmented with an additional column In_Control, which flags the control/treatment group
        
        2) Bool: True if all of the statistical test were passed, False otherwise. 
        
        3) Prints bar plot of p-values for each control_col supplied. 
    
    """
        
    pass_tests = False
    
    print "Making control group..."
    df[control_label] = df[id_col].apply(lambda x: assign_control(x, control_group_name, control_size))
        
    print "Checking control distributions..."
    pvalues = []
    for control in control_cols:
        
        #Use KS test for continue distributions
        if df[control].dtype in [np.dtype('float64'), np.dtype('int64')]:
            f = np.histogram(df[df[control_label]][control].values, bins = 100, normed = True)[0]
            g = np.histogram(df[~df[control_label]][control].values, bins = 100, normed = True)[0]
            p = stats.ks_2samp(f, g)[1]
            pvalues.append(p)
                
        #Use Chisquare for categorical 
        elif df[control].dtype in [np.dtype('object'), np.dtype('bool'), np.dtype('string')]:
            f = df[df[control_label]][control].value_counts(normalize = True).values
            g = df[~df[control_label]][control].value_counts(normalize = True).values
            p = stats.chisquare(f, g)[1]
            pvalues.append(p)
                
        else: 
            print "ERROR: Unknow dtype: ", df[control].dtype
            return -1
        
    pass_tests = np.all([p >= alpha for p in pvalues])
        
    if not pass_tests:
        for control, p in zip(control_cols, pvalues):
            if p < alpha:
                print "WARNING: ", control, ' has p-value below alpha, ', p
            
    #Plot p-values
    pd.DataFrame(pvalues, index = control_cols).plot(kind = 'bar', fontsize = 14, legend = False)
    sns.despine()
    plt.axhline(alpha, color = 'k', linestyle = '--', label = r'$\alpha =$ %0.2f' % alpha)
    #plt.legend(loc = 2, fontsize = 16)
    plt.title('P-values', fontsize = 18)
    
    return pass_tests