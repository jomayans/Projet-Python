import mlflow
import pandas as pd
import pickle
import os
import tempfile


def mlflow_fun(gmodel, experiment_name, num_app):
    mlflow.set_experiment(experiment_name=experiment_name)
    grid_params = gmodel.param_Combination
    # Sérialiser l'objet ct_data_transformer en bytes
    artifact_data = pickle.dumps(gmodel.ct_data_transformer)

# Créer un répertoire temporaire pour stocker l'objet sérialisé
    with tempfile.TemporaryDirectory() as tmpdir:
    # Écrire l'objet sérialisé dans un fichier temporaire
        artifact_path = os.path.join(tmpdir, "ct_data_transformer.pkl")
        with open(artifact_path, "wb") as f:
            f.write(artifact_data)

    # Envoyer le fichier temporaire en tant qu'artefact à MLflow
        mlflow.log_artifacts(tmpdir, artifact_path="ct_data_transformers")
    
    for idx, params in enumerate(grid_params) :
        run_name = f"run_{idx}"
        # Si une exécution est déjà active, terminer la précédente
        if mlflow.active_run():
               mlflow.end_run()
        # Démarrer une nouvelle exécution
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

            # Train the model
            
            gmodel.train_model(params=params)



            # Log fit metrics
            metrics = gmodel.evaluate_model()
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model as an artifact
            model_artifact_path = f"model_{run_name}.pkl"
            #mlflow.log_artifact(model_artifact_path, artifact_path="models")
            mlflow.sklearn.log_model(gmodel.model, model_artifact_path)  # Log the trained model

            # Log training data URL
            mlflow.log_param("num_appli", num_app)






