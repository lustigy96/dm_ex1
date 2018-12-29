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


#not good for our goal
def LOG_implementation(train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E):
    # f test on linear regression

    curr_df = train_frame.loc[:, ["id", sagnificant_x]]  # from coreallation scatter graphs
    for c in stat_vec["NAME"]:
        if c == sagnificant_x: continue
        temp_df = pd.merge(curr_df, train_frame.loc[:, ["id", c]], how='outer')
        curr_var, model1 = dm_tools.log(curr_df.drop(["id"], axis=1), train_frame[predict_E])
        test_var, model2 = dm_tools.log(temp_df.drop(["id"], axis=1), train_frame[predict_E])
        curr_df, model = statistics.z_test(curr_var, test_var, defines.alpha_f, relevant_vec, c, model1, model2,
                                           temp_df, curr_df)
    if len(test_frame)==0 :
        predictions = []
    else:
        predictions = model.predict(test_frame.loc[:, relevant_vec])
    return predictions, model

def LM_implementation(train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E):

    curr_df = train_frame.loc[:, ["id", sagnificant_x]]  # from coreallation scatter graphs
    for c in stat_vec["NAME"]:
        if c == sagnificant_x: continue
        temp_df = pd.merge(curr_df, train_frame.loc[:, ["id", c]], how='outer')
        curr_var, model1 = dm_tools.linearReg(curr_df.drop(["id"], axis=1), train_frame[predict_E])
        test_var, model2 = dm_tools.linearReg(temp_df.drop(["id"], axis=1), train_frame[predict_E])
        # curr_df, model=statistics.f_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)
        curr_df, model = statistics.z_test(curr_var, test_var, defines.alpha_f, relevant_vec, c, model1, model2,
                                           temp_df, curr_df)
    if len(test_frame)==0 :
        predictions = []
    else:
        predictions = model.predict(test_frame.loc[:, relevant_vec])
    return predictions, model

def POLY_implementation(deg,train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E):
    # f test on linear regression
    curr_df=train_frame.loc[:,["id",sagnificant_x]] #from coreallation scatter graphs
    for c in stat_vec["NAME"]:
        if c==sagnificant_x: continue
        temp_df = pd.merge(curr_df,train_frame.loc[:,["id",c]],how='outer')
        curr_var, model1 = dm_tools.poly(curr_df.drop(["id"],axis=1), train_frame[predict_E],deg=deg)
        test_var, model2 = dm_tools.poly(temp_df.drop(["id"],axis=1), train_frame[predict_E], deg=deg)
        curr_df, model=statistics.z_test(curr_var,test_var,defines.alpha_f,relevant_vec,c,model1,model2,temp_df,curr_df)
    poly=PolynomialFeatures(deg)
    if len(test_frame)==0: predictions=[]
    else:
        X_test_transform = poly.fit_transform(test_frame.loc[:,relevant_vec])
        predictions= model.predict(X_test_transform)
    return predictions, model

def LASSO_implementation(alpha,train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E):
    curr_df = train_frame.loc[:, ["id", sagnificant_x]]  # from coreallation scatter graphs
    for c in stat_vec["NAME"]:
        if c == sagnificant_x: continue
        temp_df = pd.merge(curr_df, train_frame.loc[:, ["id", c]], how='outer')
        curr_var, model1 = dm_tools.lasso(curr_df.drop(["id"], axis=1), train_frame[predict_E], alpha=alpha)
        test_var, model2 = dm_tools.lasso(temp_df.drop(["id"], axis=1), train_frame[predict_E], alpha=alpha)
        curr_df, model = statistics.z_test(curr_var, test_var, defines.alpha_f, relevant_vec, c, model1, model2, temp_df, curr_df)
    if len(test_frame)==0 :
        predictions = []
    else:
        predictions = model.predict(test_frame.loc[:, relevant_vec])
    return predictions, model

def RIDGE_implementation(alpha,train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E):
    curr_df = train_frame.loc[:, ["id", sagnificant_x]]  # from coreallation scatter graphs
    for c in stat_vec["NAME"]:
        if c == sagnificant_x: continue
        temp_df = pd.merge(curr_df, train_frame.loc[:, ["id", c]], how='outer')
        curr_var, model1 = dm_tools.ridge(curr_df.drop(["id"], axis=1), train_frame[predict_E], alpha=alpha)
        test_var, model2 = dm_tools.ridge(temp_df.drop(["id"], axis=1), train_frame[predict_E], alpha=alpha)
        curr_df, model = statistics.z_test(curr_var, test_var, defines.alpha_f, relevant_vec, c, model1, model2,
                                           temp_df, curr_df)
    if len(test_frame)==0 :
        predictions = []
    else:
        predictions = model.predict(test_frame.loc[:, relevant_vec])
    return predictions,model
#----------------------------main------------------------
#plot examples: https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/


#read data and split to train and test 1800:600
if not defines.TEST_MODE:

    predict_E="E_f"
    if predict_E=="E_f":
        sagnificant_x="ang_c"
    else: sagnificant_x="x_Al"

    data_f = os.path.join(defines.PATH_TO_FILES, 'train.csv')
    data_frame=man_data.make_dataFrame(data_f)
    train_frame=data_frame.iloc[:1800, :]
    test_frame=data_frame.iloc[1800:, :]

    train_frame = train_frame.loc[train_frame['sg'] == 194]
    test_frame = test_frame.loc[test_frame['sg'] == 194]

    # dm_tools.correlation_graphs(data_frame=train_frame,Ef=1,Ebg=0) #correlation between E_f, E_bg
    stat_vec=statistics.simple_Stat(train_frame) #calc statistics - sd, ang, corelation and filter

    # # DEBUG
    # train_frame=train_frame.loc[train_frame['sg'] == 227]
    # test_frame=test_frame.loc[test_frame['sg'] == 227]
    # train_frame=train_frame.loc[train_frame['sg'] == 194]
    # test_frame=test_frame.loc[test_frame['sg'] == 194 ]


    # train_frame=data_frame.iloc[400:650, :]
    # t_1=data_frame.iloc[:400, :]
    # t_2=data_frame.iloc[650:, :]
    # test_frame=pd.concat([t_1,t_2], axis=0)
    # test_frame=data_frame.iloc[650:, :]
    # print "1"
    # print t_1
    # print test_frame

    gap=2400/10;
    s_lm, s_poly=0,0;
    # CROSS VALIDATION
    # for i in range(10):
    #     test_frame=data_frame.iloc[i*gap+1:(i+1)*gap-1, :] #10 precent
    #     train_1 =data_frame.iloc[:gap*i+1, :]
    #     train_2=data_frame.iloc[(i+1)*gap-1:, :]
    #     train_frame=pd.concat([train_1,train_2], axis=0)



    if defines.LM_BY_ANGC:
        ang_c_vec=[0,8,11,18]
        for i in range(len(ang_c_vec)):
            if i==len(ang_c_vec)-1 and (train_frame['ang_c'] > ang_c_vec[i]).any():
                curr_df = train_frame.loc[train_frame['ang_c']>ang_c_vec[i]]
                test_1=test_frame.loc[test_frame['ang_c']>ang_c_vec[i]]
            elif (train_frame['ang_c'] > ang_c_vec[i]).any():
                curr_df = train_frame.loc[train_frame['ang_c'] < ang_c_vec[i+1]];
                curr_df = curr_df.loc[curr_df['ang_c'] > ang_c_vec[i]];
                test_1 = test_frame.loc[test_frame['ang_c'] < ang_c_vec[i+1]];
                test_1 = test_1.loc[test_1['ang_c'] > ang_c_vec[i]];
            if len(np.array(curr_df["id"])) > 0 and len(np.array(test_1["id"]))>0:
                predictions, model = dm_tools.OLS_MODEL(np.array(curr_df["ang_c"]),np.array(curr_df[predict_E]))
                y=test_1.loc[:,[predict_E]]
                X=test_1.loc[:,["ang_c"]]
                print model.rsquared
                # print model.score(X,y)
    if defines.LM: #lm with z test  #f test on linear regression
        relevant_vec = [sagnificant_x]
        predictions, model=LM_implementation(train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E)
        y=test_frame.loc[:,[predict_E]]
        X=test_frame.loc[:,relevant_vec]
        print model.score(X,y)
        s_lm+= model.score(X,y)

    if defines.LOG:
        relevant_vec = [sagnificant_x]
        predictions, model =LOG_implementation(train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E)
        y=test_frame.loc[:,[predict_E]]
        X=(test_frame.loc[:,relevant_vec])
        print model.score(X,y)

    if defines.POLY:
        deg=2
        relevant_vec=[sagnificant_x]
        predictions, model=POLY_implementation(deg,train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E)
        y=test_frame.loc[:,[predict_E]]
        poly = PolynomialFeatures(deg)
        X_test_transform = poly.fit_transform(test_frame.loc[:, relevant_vec])
        s_poly+= model.score(X_test_transform,y)
        print model.score(X_test_transform,y)

    if defines.LASSO:
        alpha=0.05
        relevant_vec = [sagnificant_x]
        predictions, model=LASSO_implementation(alpha,train_frame,test_frame,relevant_vec,stat_vec,sagnificant_x,predict_E)
        y = test_frame.loc[:, [predict_E]]
        X = (test_frame.loc[:, relevant_vec])
        print model.score(X, y)

    if defines.RIDGE:
        alpha = 0.2
        relevant_vec = [sagnificant_x]
        predictions, model=RIDGE_implementation(alpha, train_frame,test_frame, relevant_vec, stat_vec,sagnificant_x,predict_E)
        y = test_frame.loc[:, [predict_E]]
        X = (test_frame.loc[:, relevant_vec])
        print model.score(X, y)

     #doesnt work
    if defines.OLS:
            relevant_vec=[sagnificant_x]
            curr_df=train_frame.loc[:,["id",sagnificant_x]] #from coreallation scatter graphs
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
    plt.plot(np.array(test_frame["id"]), predictions, 'g', np.array(test_frame["id"]), np.array(y), 'yo');
    plt.xlabel("id")
    plt.ylabel("E_bg")
    plt.title("grean for prediction and yellow for real")
    plt.show()






if defines.TEST_MODE:
    predict_E_arr=['E_f','E_bg']
    col = ['id', 'formation_energy_ev_natom', 'bandgap_energy_ev']
    predictions, predictions_1, predictions_2=[],[],[]

    train_file = os.path.join(defines.PATH_TO_FILES, 'train.csv')
    test_file = os.path.join(defines.PATH_TO_FILES, 'test.csv')
    train_frame = man_data.make_dataFrame(train_file)
    test_frame = man_data.make_dataFrame(test_file)
    writer = pd.ExcelWriter('output.xlsx');


    stat_vec = statistics.simple_Stat(train_frame)

    # sg==194 -> ridge. other -> poly
    if defines.POLY_194_RIDGE:
        #predict E_bg - normal poly:
        predict_E="E_bg"
        deg = 2
        sagnificant_x="x_Al"
        relevant_vec = [sagnificant_x]
        pre, model = POLY_implementation(deg,train_frame,[],relevant_vec,stat_vec,sagnificant_x,predict_E)
        poly = PolynomialFeatures(deg)
        X_test_transform = poly.fit_transform(test_frame.loc[:, relevant_vec])
        predictions_bg=model.predict(X_test_transform)

        # predict E_f - if sg=194->use ridge, else: use poly:
        predict_E="E_f"
        sagnificant_x = "x_Al"
        #poly part
        train_1=train_frame.loc[train_frame['sg'] != 12]
        test_1=test_frame.loc[test_frame['sg'] != 12]
        deg = 2
        relevant_vec = [sagnificant_x]
        pre, model = POLY_implementation(deg, train_1,test_1, relevant_vec, stat_vec,sagnificant_x,predict_E)
        # poly = PolynomialFeatures(deg)
        # X_test_transform = poly.fit_transform(test_1.loc[:, relevant_vec])
        predictions_1.append(pre)
        #ridge part
        train_2 = train_frame.loc[train_frame['sg'] == 12]
        test_2 = test_frame.loc[test_frame['sg'] == 12]
        alpha = 0.2
        relevant_vec = [sagnificant_x]
        pre, model = RIDGE_implementation(alpha, train_frame,test_2, relevant_vec, stat_vec,sagnificant_x,predict_E)
        predictions_2.append(model.predict(test_2.loc[:, relevant_vec]))
        #unite all
        ex_dic_bg = {
            'id': np.array(test_frame["id"]),
            'bandgap_energy_ev': predictions_bg
        }
        ex_dic_1 = {
            'id': np.array(test_1["id"]),
            'formation_energy_ev_natom': predictions_1[0],
        }
        ex_dic_2 = {
            'id': np.array(test_2["id"]),
            'formation_energy_ev_natom': predictions_2[0],
        }

        df_bg= pd.DataFrame(ex_dic_bg, columns=["id","bandgap_energy_ev"])
        df_1 = pd.DataFrame(ex_dic_1, columns=["id","formation_energy_ev_natom"]);
        df_2 = pd.DataFrame(ex_dic_2, columns=["id","formation_energy_ev_natom"]);

        df=pd.concat([df_1,df_2])
        df=df.sort_values(by=['id'])
        df=pd.merge(df, df_bg, how='outer')

    # sg==194 -> OLS. other -> poly
    if defines.POLY_194_OLS_BY_ANGC:
        # predict E_bg - normal poly:
        predict_E = "E_bg"
        deg = 2
        sagnificant_x = "x_Al"
        relevant_vec = [sagnificant_x]
        predictions_bg, model = POLY_implementation(deg, train_frame,test_frame, relevant_vec, stat_vec, sagnificant_x, predict_E)

        # predict E_f - if sg=194->use OLS, else: use poly:
        predict_E = "E_f"
        sagnificant_x = "x_Al"
        # poly part
        train_1 = train_frame.loc[train_frame['sg'] != 12]
        test_1 = test_frame.loc[test_frame['sg'] != 12]
        train_1 = train_1.loc[train_frame['sg'] != 194]
        test_1 = test_1.loc[test_frame['sg'] != 194]

        deg = 2
        relevant_vec = [sagnificant_x]
        predictions_1, model = POLY_implementation(deg, train_1,test_1, relevant_vec, stat_vec, sagnificant_x, predict_E)
        # OLS part - by ang C
        #sg12
        ang_c_vec=[0,8,11,18]
        train_2 = train_frame.loc[train_frame['sg'] == 12]
        test_2 = test_frame.loc[test_frame['sg'] == 12]
        relevant_vec = [sagnificant_x]
        booli = False
        for i in range(len(ang_c_vec)):
            a = train_2['ang_c'] > ang_c_vec[i]
            if i == len(ang_c_vec) - 1 and (train_2['ang_c'] > ang_c_vec[i]).any():
                curr_df = train_2.loc[train_2['ang_c'] > ang_c_vec[i]]
                test_part = test_2.loc[test_2['ang_c'] > ang_c_vec[i]]
            elif i < len(ang_c_vec) - 1:
                curr_df = train_2.loc[train_2['ang_c'] < ang_c_vec[i + 1]];
                curr_df = curr_df.loc[curr_df['ang_c'] > ang_c_vec[i]];
                test_part = test_2.loc[test_2['ang_c'] < ang_c_vec[i + 1]];
                test_part = test_part.loc[test_part['ang_c'] > ang_c_vec[i]];

            if len(np.array(curr_df["id"])) > 0 and len(np.array(test_part["id"])) > 0:
                pre, model = dm_tools.OLS_MODEL(np.array(curr_df["ang_c"]), np.array(curr_df[predict_E]))
                X = test_part.loc[:, ["ang_c"]]
                predictions_2 = model.predict(X)

                ex_dic_temp = {
                    'id': np.array(test_part["id"]),
                    'formation_energy_ev_natom': predictions_2
                }
                df_temp = pd.DataFrame(ex_dic_temp, columns=["id", "formation_energy_ev_natom"])
                if not booli:
                    df_12 = df_temp;
                else:
                    df_12 = pd.concat([df_12, df_temp])
                booli = True

        train_2 = train_frame.loc[train_frame['sg'] == 194]
        test_2 = test_frame.loc[test_frame['sg'] == 194]
        relevant_vec = [sagnificant_x]
        booli = False
        for i in range(len(ang_c_vec)):
            a = train_2['ang_c'] > ang_c_vec[i]
            if i == len(ang_c_vec) - 1 and (train_2['ang_c'] > ang_c_vec[i]).any():
                curr_df = train_2.loc[train_2['ang_c'] > ang_c_vec[i]]
                test_part = test_2.loc[test_2['ang_c'] > ang_c_vec[i]]
            elif i < len(ang_c_vec) - 1:
                curr_df = train_2.loc[train_2['ang_c'] < ang_c_vec[i + 1]];
                curr_df = curr_df.loc[curr_df['ang_c'] > ang_c_vec[i]];
                test_part = test_2.loc[test_2['ang_c'] < ang_c_vec[i + 1]];
                test_part = test_part.loc[test_part['ang_c'] > ang_c_vec[i]];

            if len(np.array(curr_df["id"])) > 0 and len(np.array(test_part["id"])) > 0:
                pre, model = dm_tools.OLS_MODEL(np.array(curr_df["ang_c"]), np.array(curr_df[predict_E]))
                X = test_part.loc[:, ["ang_c"]]
                predictions_3 = model.predict(X)

                ex_dic_temp_194 = {
                    'id': np.array(test_part["id"]),
                    'formation_energy_ev_natom': predictions_3
                }
                df_temp = pd.DataFrame(ex_dic_temp_194, columns=["id", "formation_energy_ev_natom"])
                if not booli:
                    df_194 = df_temp;
                else:
                    df_194 = pd.concat([df_194, df_temp])
                booli = True

    # unite all
        ex_dic_bg = {
            'id': np.array(test_frame["id"]),
            'bandgap_energy_ev': predictions_bg
        }
        ex_dic_1 = {
            'id': np.array(test_1["id"]),
            'formation_energy_ev_natom': predictions_1,
        }

        df_bg = pd.DataFrame(ex_dic_bg, columns=["id", "bandgap_energy_ev"])
        df_1 = pd.DataFrame(ex_dic_1, columns=["id", "formation_energy_ev_natom"]);

        df = pd.concat([df_1, df_194])
        df = pd.concat([df, df_12])
        df = df.sort_values(by=['id'])
        df = pd.merge(df, df_bg, how='outer')
        print df

    if defines.POLY:# f test on linear regression
        for predict_E in predict_E_arr:
            if predict_E=="E_f":
                sagnificant_x="ang_c"
            else: sagnificant_x="x_Al"
            deg = 2
            relevant_vec = [sagnificant_x]
            curr_df = train_frame.loc[:, ["id", sagnificant_x]]  # from coreallation scatter graphs
            pre, model = POLY_implementation(deg, train_frame,test_frame, relevant_vec, stat_vec, sagnificant_x, predict_E)
            predictions.append(pre)

        ex_dic = {
            'id': range(1,601),
            'formation_energy_ev_natom': predictions[0],
            'bandgap_energy_ev': predictions[1]
        }
        col=['id', 'formation_energy_ev_natom','bandgap_energy_ev']
        df = pd.DataFrame(ex_dic, columns=col);

    df.to_excel(writer, 'Sheet1')
    writer.save()
