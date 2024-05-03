import sys
sys.path.append('/home/onyxia/work/cine-insights/src/models')
sys.path.append('/home/onyxia/work/cine-insights/src/features')

import TRain 
import Preprocess_predicted_data 


class Predicto:
    def __init__(self, trained_model):
        self.model = trained_model
        self.ct_data_transformer = self.model.ct_data_transformer
    # Le reste de votre code...

    def predict(self, raw_X):
        # faire le preprocessing des nouvelles donnes
        XX_transformed = Preprocess_predicted_data.preprocessing_pipeline(raw_X, self.ct_data_transformer)
        
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné. Veuillez d'abord entraîner le modèle." )

        return self.model.predict_(XX_transformed)


