
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # sklearn.cross_validation in old versions
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


categorical = ['production_companies', 'production_countries', 'spoken_languages', 'keywords']
cols_to_binerize = ['homepage', 'status', 'spoken_languages', 'Keywords']
cols_to_count_values = ['production_countries', 'production_companies', 'spoken_languages', 'keywords', 'cast', 'crew']
# Formater la date pour obtenir le jour, le mois et l'année seulement
today = datetime(2024, 4, 14)
date_col = "release_date"
Cols_to_Remove = ['Keywords', 'spoken_languages','homepage', 'production_countries','production_companies', 'release_date', 'poster_path', 'id', 'status','imdb_id', 'logRevenue', 'logBudget',"released"]
log_num_feats = ["budget", "popularity"]
#num_feats = ["Duration"] # non requis, à explorer ultérieurement
#cat_feats = ['belongs_to_collection', 'has_homepage', 'released']
cat_feats = [ 'has_homepage','release_month','release_year']

#encode_feats = genre_feature_name(df)

hash_feats = ['production_countries_count', 'production_companies_count', 'spoken_languages_count','keyword_count']

#hash_feats = ['production_countries_count', 'production_companies_count', 'spoken_languages_count',
             #'keyword_count', 'cast_count', 'crew_count']



def shuffle_split(df, scale=0.25, target="revenue"):   
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_remind, y_train, y_remind = train_test_split(X, y, train_size=(1-scale), random_state=42)
    X_val, X_test,  y_val, y_test = train_test_split(X_remind, y_remind, test_size=0.8, random_state=42)
    # 0.8 = 80% des données restantes pour le jeu de test et les 20% pour la validation

    print("(X.shape, y.shape )","\n")  # Correction ici
    print("Pour le train :", X_train.shape, y_train.shape,"\n")
    print("Pour le test :" ,X_test.shape, y_test.shape,"\n")
    print("Pour la validation :", X_val.shape, y_val.shape,"\n")

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def Binarizer(df, cols_to_binarize):
    for col in cols_to_binarize:
        num_col_name = 'num_' + col
        df[num_col_name] = 1
        df.loc[pd.isnull(df[col]), num_col_name] = 0

        print("Binerization well done")

# Définir la fonction count_strings


def count_strings(s):
    """
    Cette fonction compte le nombre de sous-chaînes séparées par des virgules dans une chaîne donnée.
    
    Args:
    - s (str): La chaîne de caractères à analyser.
    
    Returns:
    - int: Le nombre de sous-chaînes, ou NaN si la chaîne est nulle.
    """
    # Vérifier si la chaîne est nulle
    if pd.isna(s):
        return np.nan
    
    # Diviser la chaîne en fonction des virgules et compter le nombre de sous-chaînes obtenues
    return len(s.split(','))


def apply_count(df, cols_to_count_values):
    for col in cols_to_count_values:
        new_col_name = col + "_count"
        df[new_col_name] = df[col].apply(count_strings)

def remove_empty_date_line(X_df,Y_df,date_col=date_col):
    X_df,Y_df=X_train[X_df.date_col.isnull()==False],Y_df[X_df.date_col.isnull()==False]

    return X_df,Y_df

# extraire l'année à 4 chiffres à partir d'une chaîne représentant une date
def yearfix(x): # run year fix, then date fix
    
    r = x[:4]
    return int(r)
def apply_yearfix(df,date_col=date_col,col_name="release_year"):
    # 
     df[col_name] = df[date_col].apply(lambda x: apply_yearfix(x))

def monthfix(x): # run year fix, then date fix
    r = x[5:7]
    return int(r)

def monthfix(df,date_col=date_col,col_name="release_month"):
    # 
     df[col_name] = xtn[date_col].apply(lambda x: apply_yearfix(x))




def str_to_datetime(str_date, today):
    # Convertir la chaîne en objet datetime
    date_reference = datetime.strptime(str_date, "%Y-%m-%d")
    # Calculer la différence entre la date de référence et la date actuelle
    difference = today - date_reference

    return round(difference.total_seconds() / (3600 * 24), 5)

with_duration=False
def add_duration_col(df, with_duration=False, date_col="release_date"):
    if with_duration:
        today = datetime(2024, 4, 14)  # Remplacer cette date par la date actuelle
        df["Duration"] = df[date_col].apply(lambda x: str_to_datetime(x, today))

#
def model_features(df, cols_to_remove=Cols_to_Remove):
    return list(set(df.columns) - set(cols_to_remove))

## apply Log function to features durint the preprocessing
class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)
    

# pipiline 

def gender_preprocessing(df):
    # set a liste containing all movie genders 
    X_all_genres = df['genres'].str.split(', ', expand=True)
    # Convertir chaque genre en colonnes binaires
    X_all_genres = pd.get_dummies(X_all_genres.apply(lambda x: pd.Series(x).str.strip()), prefix='', prefix_sep='').groupby(level=0, axis=1).sum()
    
    # Renommer les colonnes binaires pour correspondre aux noms de colonnes du premier code
    X_all_genres.columns = ['genre_' + col.replace(' ', '_') for col in X_all_genres.columns]
    # Ajouter les colonnes binaires encodées en one-hot au DataFrame complet
    X_all = pd.concat([df, X_all_genres], axis=1)

    return X_all


def genre_feature_name(df):
    cols = df.columns
# Séquence à rechercher
    sequence = 'genre_'

# Filtrer les éléments de la liste qui commencent par la séquence
    return [element for element in cols if element.startswith(sequence)]


def log_pipeline():
    log_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
        ('logger', Log1pTransformer()),
        ('scaler', StandardScaler())
    ])

    return log_pipe


def num_pipeline():
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    return num_pipeline


def date_pipeline():
    date_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='median'))
    ])

    return date_pipeline


def cat_pipeline():
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # Ajoutez d'autres étapes au besoin
    ])

    return cat_pipeline


def encode_pipeline():
    encode_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    return encode_pipeline


def hash_pipeline():
    hash_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    return hash_pipeline


def preprocessing_pipeline(df):
    # initialisation de variable
    encode_feats = genre_feature_name(df)

    # initialisation de transormation
    log_pipe = log_pipeline()
    cat_pipe = cat_pipeline()
    hash_pipe = hash_pipeline()
    encode_pipe = encode_pipeline()

    # Appliquer le pipeline de prétraitement et obtenir les données transformées
    transformed_data = ColumnTransformer([
        # Transformation des caractéristiques numériques avec log
        ('log_num_feats', log_pipe, log_num_feats),
        
        # Transformation des caractéristiques catégorielles
        ('cat_feats', cat_pipe, cat_feats),
        
        # Encodage des caractéristiques catégorielles
        ('encode_feats', encode_pipe, encode_feats),
        
        ('hash_feats', hash_pipe, hash_feats)
        # Ajoutez d'autres transformations au besoin
    ], remainder='drop', n_jobs=-1).fit_transform(df)

    # Créer un DataFrame à partir des données transformées
    transformed_df = pd.DataFrame(transformed_data, columns=df.columns)

    return transformed_df



