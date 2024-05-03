import pandas as pd
import IPython.display as display

import boto3
import pyarrow.parquet as pq
# Exemple d'utilisation
minio_url = 'URL_VERS_MINIO'
access_key = 'VOTRE_ACCESS_KEY'
secret_key = 'VOTRE_SECRET_KEY'
bucket_name = 'NOM_DU_BUCKET'
object_key = 'CHEMIN_VERS_LE_FICHIER_DANS_LE_BUCKET'

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



parquet_table = read_parquet_from_minio(minio_url, access_key, secret_key, bucket_name, object_key)
if parquet_table is not None:
    # Affichage des premières lignes du tableau Parquet
    print(parquet_table.to_pandas().head())



        
    return df
