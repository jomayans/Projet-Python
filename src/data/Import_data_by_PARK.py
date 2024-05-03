import pandas as pd
import IPython.display as display

import boto3
import pyarrow.parquet as pq


def read_parquet_from_minio(minio_url, access_key, secret_key, bucket_name, object_key):
    # Connexion à MinIO
    s3 = boto3.client('s3', endpoint_url=minio_url, aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    try:
        # Téléchargement du fichier Parquet depuis le bucket MinIO
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        # Lecture du fichier Parquet
        table = pq.read_table(response['Body'])
        return table
    except Exception as e:
        print(f"Une erreur s'est produite lors de la lecture du fichier Parquet : {e}")
        return None




