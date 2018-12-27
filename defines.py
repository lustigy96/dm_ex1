#DEFINES

PATH_TO_FILES= "C:\YAEL\BGU\data_for_proj\kernel_v0\\"


#tool:
LM=         False #linear regression
LM_BY_ANGC= False #linear regression only on ang_c , with areas separation
OLS=        False #Ordinary Least Squares
LOG=        False #log regression
POLY=       True #polynomial regression
LASSO=      False #lasso regression
RIDGE=      False #ridge resression

POLY_194_RIDGE=         False#sg==194 -> ridge. other -> poly
POLY_194_OLS_BY_ANGC=   False#sg==194 -> ols. other -> poly

TEST_MODE = False  # do we work or making a test submission?

alpha_f = 0.05 #f test
SD2AVG_TRESH=0.2