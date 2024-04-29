import mlflow
import pandas as pd

def mlflow_fun(gmodel, experiment_name, num_appli, evaluation_data):
    mlflow.set_experiment(experiment_name=experiment_name)
    grid_params = gmodel.param_Combination
    k = len(grid_params)
    
    for run_index in range(k):
        run_name = f"run_{run_index}"
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            params = grid_params[run_index]
            for param in params:
                mlflow.log_param(param, params[param])

            # Train the model
            gmodel._train_model(**params.values)

            # Log fit metrics
            metrics = gmodel.evaluate_model()
            for score, value in metrics.items():
                mlflow.log_metric(score, value)

            # Log model as an artifact
            model_path = f"model_{run_name}.pkl"
            gmodel.save_model(model_path)
            mlflow.log_artifact(model_path, artifact_path="models")

            # Make predictions on evaluation data
            predictions = gmodel.predict_(evaluation_data)

            # Save predictions as an artifact
            predictions_df = pd.DataFrame({"Prediction": predictions})
            predictions_file_path = f"predictions_{run_name}.csv"
            predictions_df.to_csv(predictions_file_path, index=False)
            mlflow.log_artifact(predictions_file_path, artifact_path="predictions")

            # Log training data URL
            mlflow.log_param("num_appli", num_appli)
