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
import OUR_INPUT_Data_Preprocessor         
from OUR_INPUT_Data_Preprocessor import * 
import  OUR_Trainer_  
from OUR_Trainer_ import *
import OUR_Data_TO_Predict_Preprocessor  # requirements train_model_xgboot22a6 and  Build_Feats1a
from  OUR_Data_TO_Predict_Preprocessor import *  #  Utilisez les fonctions du module file
import OUR_Predictor_  # requirements train_model_xgboot22a6 and  Build_Feats1a
from  OUR_Predictor_ import *  #  Utilisez les fonctions du module file
filename="Raw_df_50_ligne.csv"
path_dir="/home/onyxia/work/cine-insights/src/data/"
python_inter_pretor=""

raw_df=pd.read_csv(path_dir+filename)
#(X_train,y_train),(X_test,y_test),(X_val,y_val),ct_data_transformer=BON_Predict_model_A210.preprocessing_data(raw_df)
model=OUR_Trainer_.XGBRegressorWrapper(raw_df)
model.display_results()

