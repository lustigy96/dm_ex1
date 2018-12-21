#!/usr/bin/env python
import defines
import  plots
import  man_data
import dm_tools
import statistics

import pandas as pd     #read csv files
import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy
import statsmodels.api as sm


#----------------------------main------------------------
TEST_MODE = False  # do we work or making a test submission?

#read data and split to train and test 1800:600
if not TEST_MODE:
    #plot examples: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
    data_f = os.path.join(defines.PATH_TO_FILES, 'train.csv')
    data_frame=man_data.make_dataFrame(data_f)
    train_frame=data_frame.iloc[:1800, :]
    test_frame=data_frame.iloc[1800:, :]

    dm_tools.correlation_graphs(data_frame=train_frame,Ef=1,Ebg=1) #correlation between E_f, E_bg
    stat_vec=statistics.simple_Stat(train_frame) #calc statistics - sd, ang, corelation and filter

    # train_frame=data_frame.iloc[400:650, :]
    # t_1=data_frame.iloc[:400, :]
    # t_2=data_frame.iloc[650:, :]
    # test_frame=pd.concat([t_1,t_2], axis=0)
    # test_frame=data_frame.iloc[650:, :]
    # print "1"
    # print t_1
    # print test_frame
    predict_E="E_f"
    gap=2400/10;
    s_lm, s_poly=0,0;
    # for i in range(10):
    #     test_frame=data_frame.iloc[i*gap+1:(i+1)*gap-1, :] #10 precent
    #     train_1 =data_frame.iloc[:gap*i+1, :]
    #     train_2=data_frame.iloc[(i+1)*gap-1:, :]
    #     train_frame=pd.concat([train_1,train_2], axis=0)

    if defines.LM: #lm with z test
        #f test on linear regression
        relevant_vec=["x_Al"]
        curr_df=train_frame.loc[:,["id","x_Al"]] #from coreallation scatter graphs
        for c in stat_vec["NAME"]:
            if c=="x_Al": continue
            temp_df = pd.merge(curr_df,train_frame.loc[:,["id",c]],how='outer')
            curr_var, model1 = dm_tools.linearReg(curr_df.drop(["id"],axis=1), train_frame[predict_E])
            test_var, model2 = dm_tools.linearReg(temp_df.drop(["id"],axis=1), train_frame[predict_E])
            # curr_df, model=statistics.f_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)
            curr_df, model=statistics.z_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)
        # print test_frame
        # print relevant_vec
        predictions= model.predict(test_frame.loc[:,relevant_vec])
        y=test_frame.loc[:,[predict_E]]
        X=(test_frame.loc[:,relevant_vec])
        # print model.score(X,y)
        s_lm+= model.score(X,y)

    if defines.LOG:
        #f test on linear regression
        relevant_vec=["x_Al"]
        curr_df=train_frame.loc[:,["id","x_Al"]] #from coreallation scatter graphs
        for c in stat_vec["NAME"]:
            if c=="x_Al": continue
            temp_df = pd.merge(curr_df,train_frame.loc[:,["id",c]],how='outer')
            curr_var, model1 = dm_tools.log(curr_df.drop(["id"],axis=1), train_frame[predict_E])
            test_var, model2 = dm_tools.log(temp_df.drop(["id"],axis=1), train_frame[predict_E])
            curr_df, model=statistics.z_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)
        predictions= model.predict(test_frame.loc[:,relevant_vec])
        y=test_frame.loc[:,[predict_E]]
        X=(test_frame.loc[:,relevant_vec])
        print model.score(X,y)

    if defines.POLY:
        #f test on linear regression
        deg=2
        relevant_vec=["x_Al"]
        curr_df=train_frame.loc[:,["id","x_Al"]] #from coreallation scatter graphs
        for c in stat_vec["NAME"]:
            if c=="x_Al": continue
            temp_df = pd.merge(curr_df,train_frame.loc[:,["id",c]],how='outer')
            curr_var, model1 = dm_tools.poly(curr_df.drop(["id"],axis=1), train_frame[predict_E],deg=deg)
            test_var, model2 = dm_tools.poly(temp_df.drop(["id"],axis=1), train_frame[predict_E], deg=deg)
            curr_df, model=statistics.z_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)

        poly=PolynomialFeatures(deg)
        X_test_transform = poly.fit_transform(test_frame.loc[:,relevant_vec])
        predictions= model.predict(X_test_transform)
        y=test_frame.loc[:,[predict_E]]
        print model.score(X_test_transform,y)
        s_poly+= model.score(X_test_transform,y)
        # if(s_poly<0):
        #     print i

    if defines.LASSO:
        alpha=0.001
        relevant_vec = ["x_Al"]
        curr_df = train_frame.loc[:, ["id", "x_Al"]]  # from coreallation scatter graphs
        for c in stat_vec["NAME"]:
            if c=="x_Al": continue
            temp_df = pd.merge(curr_df,train_frame.loc[:,["id",c]],how='outer')
            curr_var, model1 = dm_tools.lasso(curr_df.drop(["id"],axis=1), train_frame[predict_E],alpha=alpha)
            test_var, model2 = dm_tools.lasso(temp_df.drop(["id"],axis=1), train_frame[predict_E], alpha=alpha)
            curr_df, model=statistics.z_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)

        predictions = model.predict(test_frame.loc[:, relevant_vec])
        y = test_frame.loc[:, [predict_E]]
        X = (test_frame.loc[:, relevant_vec])
        print model.score(X, y)

    if defines.RIDGE:
        alpha = 0.4
        relevant_vec = ["x_Al"]
        curr_df = train_frame.loc[:, ["id", "x_Al"]]  # from coreallation scatter graphs
        for c in stat_vec["NAME"]:
            if c == "x_Al": continue
            temp_df = pd.merge(curr_df, train_frame.loc[:, ["id", c]], how='outer')
            curr_var, model1 = dm_tools.ridge(curr_df.drop(["id"], axis=1), train_frame[predict_E], alpha=alpha)
            test_var, model2 = dm_tools.ridge(temp_df.drop(["id"], axis=1), train_frame[predict_E], alpha=alpha)
            curr_df, model = statistics.z_test(curr_var, test_var, defines.alpha_f, relevant_vec, c, model1, model2,
                                               temp_df, curr_df)

        predictions = model.predict(test_frame.loc[:, relevant_vec])
        y = test_frame.loc[:, [predict_E]]
        X = (test_frame.loc[:, relevant_vec])
        print model.score(X, y)

     #doesnt work
    if defines.OLS:
            relevant_vec=["x_Al"]
            curr_df=train_frame.loc[:,["id","x_Al"]] #from coreallation scatter graphs
            for c in stat_vec["NAME"]:
                temp_df = pd.merge(curr_df,train_frame.loc[:,["id",c]],how='outer')
                curr_var, model1 = dm_tools.OLS_MODEL(curr_df.drop(["id"],axis=1), train_frame[predict_E])
                test_var, model2 = dm_tools.OLS_MODEL(temp_df.drop(["id"],axis=1), train_frame[predict_E])
                A1 = np.identity(len(model1.params)); A1 = A1[1:,:]
                A2 = np.identity(len(model2.params)); A2 = A2[1:,:]
                print A1
                print A2
                print model1.f_pvalue(A1)
                if model1.f_pvalue(A1)> model2.f_pvalue(A2): #we want p_val->0 to accept the hyposesis of R^2=0
                    curr_df=temp_df
                    relevant_vec.append(c)
                    print "f::"+c+" is helpping"
                else:
                    print "f:: drop " +c

            y=test_frame.loc[:,["E_f"]]
            X=(test_frame.loc[:,relevant_vec])
            predictions= model.predict(X)
            print model.rsquared

    # print "----linear:----"
    # print (s_lm/10)
    # print "----poly:----"
    # print (s_poly/10)
    # plt.show()

if TEST_MODE:
    x_al="percent_atom_al"
    predict_E_arr=['E_f','E_bg']
    predictions=[]
    i=0;

    writer = pd.ExcelWriter('output.xlsx');

    train_file = os.path.join(defines.PATH_TO_FILES, 'train.csv')
    test_file = os.path.join(defines.PATH_TO_FILES, 'test.csv')
    train_frame = man_data.make_dataFrame(train_file)
    test_frame = man_data.make_dataFrame(test_file)

    stat_vec = statistics.simple_Stat(train_frame)

    for predict_E in predict_E_arr:
        if defines.POLY:# f test on linear regression
            deg = 2
            relevant_vec = ["x_Al"]
            curr_df = train_frame.loc[:, ["id", "x_Al"]]  # from coreallation scatter graphs
            for c in stat_vec["NAME"]:
                if c == "x_Al": continue
                temp_df = pd.merge(curr_df, train_frame.loc[:, ["id", c]], how='outer')
                curr_var, model1 = dm_tools.poly(curr_df.drop(["id"], axis=1), train_frame[predict_E], deg=deg)
                test_var, model2 = dm_tools.poly(temp_df.drop(["id"], axis=1), train_frame[predict_E], deg=deg)
                curr_df, model = statistics.z_test(curr_var, test_var, defines.alpha_f, relevant_vec, c, model1, model2,
                                                   temp_df, curr_df)

            poly = PolynomialFeatures(deg)
            X_test_transform = poly.fit_transform(test_frame.loc[:, relevant_vec])
            predictions.append(model.predict(X_test_transform))

    ex_dic = {
        'id': range(1,601),
        'formation_energy_ev_natom': predictions[0],
        'bandgap_energy_ev': predictions[1]
    }
    col=['id', 'formation_energy_ev_natom','bandgap_energy_ev']
    df = pd.DataFrame(ex_dic, columns=col);
    df.to_excel(writer, 'Sheet1')

    writer.save()
