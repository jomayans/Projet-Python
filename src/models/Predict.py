import os
import mlflow
import pandas as pd
import pickle
import yaml
import src.data.Import_data as Import_data
import src.features.preprocessing_predicted_data as preprocessing_predicted_data
import src.features.preprocess_train_data as preprocess_train_data
import mlflow.pyfunc
import re
import mlflow.sklearn


def filter_params_indices(series):
    filtered_indices = series.index[series.index.str.startswith("params.")]
    filtered_series = series.loc[filtered_indices]
    return filtered_series


def get_best_model(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    best_run = runs.loc[runs["metrics.Test RMSLE"].idxmin()]

    print("Best run:")
    print(best_run)

    best_params = filter_params_indices(best_run)
    model_artifact_uri = best_run["artifact_uri"] + "/model"

    return best_params, model_artifact_uri


def load_model(artifact_uri):
    model_path = os.path.join(artifact_uri, "model.pkl")
    return mlflow.sklearn.load_model(model_path)




yaml_file_path = "mlflow_predict.yaml"

with open(yaml_file_path, "r") as yaml_file:
        workflow = yaml.safe_load(yaml_file)

container_args = ["run_id=your_run_id", "lien=your_link"]

lien = container_args[1].split("=")[1]
run_id = container_args[0].split("=")[1]
local_dir = "../data/Download_Artefact"

data = pd.read_parquet(lien)[0:50]

if mlflow.active_run():
    mlflow.end_run()

mlflow_server_uri = container_args[4].split("=")[1]
mlflow_client = mlflow.tracking.MlflowClient()
mlflow_client.download_artifacts(run_id=run_id, path="ct_data_transformers", dst_path=local_dir)

artifact_path = os.path.join(local_dir, "ct_data_transformer.pkl")

with open(artifact_path, "rb") as f:
    ct_data_transformer = pickle.load(f)

experiment_name = "Final Experiement 06 mai version2"

def predictor(data=data,experiment_name=experiment_name):
    X_transform = preprocessing_predicted_data.preprocessing_pipeline(data, ct_data_transformer, with_duration=False)
    transformed_data = ct_data_transformer.transform(X_transform)
    transformed_data = pd.DataFrame(transformed_data, columns=X_transform.columns)
    best_params, model_artifact_uri = get_best_model(experiment_name)
    best_model = load_model(model_artifact_uri)
    print("well done")
    return best_model.predict(transformed_data)
