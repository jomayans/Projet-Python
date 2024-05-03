import mlflow
import pandas as pd
import pickle
import src.features.preprocess_train_data
import src.features.preprocessing_predicted_data
import src.models.train_models
import src.models.Evaluation
import src.data.Import_data


def load_data(in_path, name, n_display=1, show_info=False, nrows=None):
    df = pd.read_parquet(in_path, nrows=nrows)
    print(f"{name}: shape is {df.shape}")
    df = df.rename(columns={"keywords": "Keywords"})

    if show_info:
        print(df.info())

    if n_display > 0:
        display.display(df.head(n_display))

    return df

import mlflow

# Si une exécution est déjà active, terminer la précédente


# Démarrer une nouvelle exécution
    # Votre code ici

def mlflow_fun(gmodel, experiment_name, num_appli, evaluation_data):
    mlflow.set_experiment(experiment_name=experiment_name)
    grid_params = gmodel.param_Combination
     # Sauvegarder le transformateur de données une seule fois
    with open("ct_data_transformer.pkl", "wb") as f:
        pickle.dump(gmodel.ct_data_transformer, f)
    mlflow.log_artifact("ct_data_transformer.pkl", artifact_path="ct_data_transformers")
    
    for idx, params in enumerate(grid_params) :
        run_name = f"run_{idx}"
        if mlflow.active_run():
               mlflow.end_run()
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
            gmodel.save_best_model(model_artifact_path)
            mlflow.log_artifact(model_artifact_path, artifact_path="models")
            mlflow.sklearn.log_model(gmodel.model, model_artifact_path)  # Log the trained model

            # Make predictions on evaluation data
            predictions = gmodel.predict_(evaluation_data)

            # Save predictions as an artifact
            predictions_df = pd.DataFrame({"Prediction": predictions})
            predictions_file_path = f"predictions_{run_name}.csv"
            predictions_df.to_csv(predictions_file_path, index=False)
            mlflow.log_artifact(predictions_file_path, artifact_path="predictions")

            mlflow.log_table(data=predictions_df, artifact_file=run_name+"predictions_df.json")
            # Log training data URL
            mlflow.log_param("num_appli", num_appli)
