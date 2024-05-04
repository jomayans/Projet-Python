import mlflow
import pandas as pd
import pickle
"import src.models.train_models as train_models"
import src.models.Train_testa as train_models
#import src.data.Import_data as Import_Data





def mlflow_fun(gmodel, experiment_name, num_app):
    mlflow.set_experiment(experiment_name=experiment_name)
    grid_params = gmodel.param_Combination
     # Sauvegarder le transformateur de données une seule fois
    with open("ct_data_transformer.pkl", "wb") as f:
        pickle.dump(gmodel.ct_data_transformer, f)
    mlflow.log_artifact("ct_data_transformer.pkl", artifact_path="ct_data_transformers")
    
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






