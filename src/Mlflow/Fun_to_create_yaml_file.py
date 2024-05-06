import yaml

def create_argo_yaml(filename, run_id, link, model_name, model_version):
    """
    Create an Argo YAML file with the given parameters.

    Args:
    - filename (str): Name of the YAML file to create.
    - run_id (str): MLflow run ID.
    - link (str): Link to the data source.
    - model_name (str): Name of the model.
    - model_version (int): Version of the model.
    """
    workflow = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": "my-workflow-"
        },
        "spec": {
            "entrypoint": "my-entrypoint",
            "templates": [
                {
                    "name": "my-entrypoint",
                    "container": {
                        "image": "docker-image",
                        "command": ["my-command"],
                        "args": [
                            f"--run_id={run_id}",
                            f"--link={link}",
                            f"--model_name={model_name}",
                            f"--model_version={model_version}"
                        ]
                    }
                }
            ]
        }
    }

    with open(filename, "w") as f:
        yaml.dump(workflow, f)

# Example usage
run_id = "e3bad03bcc63472d8ff9ff143af87c4e"
link = "https://minio.lab.sspcloud.fr/alimane/diffusion/TMDB_movies.parquet"
model_name = "BEST_MODEL"
model_version = 1

create_argo_yaml("mlflow_predict.yaml", run_id, link, model_name, model_version)
