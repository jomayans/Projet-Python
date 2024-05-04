
import mlflow
import pandas as pd
import src.data.Import_data as Import_data
import src.features.preprocessing_predicted_data as preprocessing_predicted_data
import src.features.preprocess_train_data as preprocess_train_data
import pickle
import re
import mlflow.pyfunc
import pickle

import yaml

yaml_file_path = "../models/mlflow_predict.yaml"


with open(yaml_file_path, "r") as yaml_file:
    workflow = yaml.safe_load(yaml_file)

# Extraire les valeurs des paramètres
generateName = workflow["metadata"]["generateName"]
entrypoint = workflow["spec"]["entrypoint"]
container_args = workflow["spec"]["templates"][0]["container"]["args"]
run_id = container_args[0].split("=")[1]
lien = container_args[1].split("=")[1]
model_name = container_args[2].split("=")[1]
model_version = container_args[3].split("=")[1]
image = workflow["spec"]["templates"][0]["container"]["image"]

# data qui ne contient pas la colonne "revenue" et qui aprtagent les même colonnes que le dataset qu'AYMAN me fournira

data = pd.read_parquet(lien)[0:50]
data.shape

loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# Définir le répertoire local où vous souhaitez télécharger l'artefact
local_dir = "../models/Download_Artefact"

# Télécharger l'artefact depuis MLflow
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_client.download_artifacts(run_id=run_id, path="ct_data_transformers", dst_path=local_dir)

# Charger l'objet à partir du fichier téléchargé
artifact_path = os.path.join(local_dir, "ct_data_transformer.pkl")
with open(artifact_path, "rb") as f:
    ct_data_transformer = pickle.load(f)

# Transformer les données afin d'avoir les  bonnes colonnes attendus par le modéle
X_transform = Preprocessing_PredictedData.preprocessing_pipeline(data, ct_data_transformer, with_duration=False)
# appliquer l'ensemble de pipeline necessaire comme celle dans l'entrainement
transformed_data = ct_data_transformer.transform(X_transform)
transformed_data = pd.DataFrame(transformed_data,columns=X_transform.columns)

"""
# Predict on a Pandas DataFrame. loaded_model.predict(data) """
def do_():
    print("well done")
    return loaded_model.predict(transformed_data)

    
