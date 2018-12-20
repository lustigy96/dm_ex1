import defines

import matplotlib.pyplot as plt
import plots
import statsmodels.api as sm
from sklearn import linear_model
import numpy as np
import scipy
from statsmodels.stats import weightstats as stests


#calc statistics - sd, ang, corelation and put in a file AND filter low variation variables
def simple_Stat(data_frame):
    stat_f = open(defines.PATH_TO_FILES + "data_stat.txt", "w")
    stat_vec = {"NAME": [], "SD": [], "AVG": [], "COR_EF": [], "COR_EB": []}
    stat_f.write("\t\tsd\t\tavg\t\tcoeff_Ef\tcoeff_Ebg\n")

    for v in data_frame:
        if v=="id": continue #or v=="sg"
        if v=="ang_alpha" or v=="ang_beta" or v=="ang_gamma": stat_f.write(v+":\t")
        else: stat_f.write(v+":\t\t")

        sd=np.std(data_frame[v].values)
        avg=np.average(data_frame[v].values)
        cor_ef=np.corrcoef(data_frame[v].values, data_frame["E_f"])[0,1]
        cor_ebg=np.corrcoef(data_frame[v].values, data_frame["E_bg"])[0,1]

        if sd/avg >defines.SD2AVG_TRESH or v=="sg":
            if v!= "E_bg" and v!="E_f":
                stat_vec["NAME"].append(v)
                stat_vec["SD"].append(sd)
                stat_vec["AVG"].append(avg)
        else: print "var_drop: "+v
        stat_f.write(format(sd,".3")+"\t\t"+format(avg,".3")+"\t\t"+format(cor_ef,".3")+"\t\t"+format(cor_ebg,".3")+"\n")
    stat_f.close()
    return stat_vec

def f_test(curr,test,alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df):
    F = np.var(curr) / np.var(test) #.values
    new_df=curr_df
    Ftest_df1= len(curr)-1
    Ftest_df2= len(test)-1
    p_value = scipy.stats.f.cdf(F, Ftest_df1, Ftest_df2)
    if p_value>alpha_f: #this is an important criteria
        new_df=temp_df
        relevant_vec.append(c)
        model=model2
        print "f:: " +c +" is important"
    else:
        model=model1
        print "f:: drop "+ c
    return new_df, model


def z_test(curr,test,alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df):
    new_df=curr_df
    avg1=np.average(curr)
    avg2=np.average(test)

    p_value = stests.ztest(curr,test,avg1-avg2,"two-sided")
    if p_value>alpha_f: #this is an important criteria, not equivalent
        new_df=temp_df
        relevant_vec.append(c)
        model=model2
        print "z:: " +c +" is important"
    else:
        model=model1
        print "z:: drop "+ c
    return new_df, model