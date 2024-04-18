import warnings

# Ignorer tous les avertissements
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/onyxia/work/cine-insights/src/models')
sys.path.append('/home/onyxia/work/cine-insights/src/data')
sys.path.append('/home/onyxia/work/cine-insights/src/features')

import pandas as pd
import joblib
import numpy as np

 #
import BON_BUILD10_1A         
from BON_BUILD10_1A import * 
#
import  BON_Model_TRAIN_A  
from BON_Model_TRAIN_A import *
#
import BON_Predict_model_A2  # requirements train_model_xgboot22a6 and  Build_Feats1a
from  BON_Predict_model_A2 import *  #  Utilisez les fonctions du module file

import BON_BUILD_FEATS_1  # requirements train_model_xgboot22a6 and  Build_Feats1a
from  BON_BUILD_FEATS_1 import *  #  Utilisez les fonctions du module file

filename="Raw_df_50_ligne.csv"
path_dir="/home/onyxia/work/cine-insights/src/data/"

raw_df=pd.read_csv(path_dir+filename)
(X_train,y_train),(X_test,y_test),(X_val,y_val),ct_data_transformer=Predict_model_A.preprocessing_data(raw_df)
model=BON_Model_TRAIN_A.XGBRegressorWrapper(raw_df)

