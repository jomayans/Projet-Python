import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Colonnes spécifiques
# Colonnes spécifiques
cols_to_count_values = ["spoken_languages","production_countries","production_companies","Keywords"] # + cats +crew
date_col = "release_date"
Cols_to_Remove = ['Keywords', 'spoken_languages','homepage', 'production_countries','production_companies', 'release_date', 'poster_path', 'id', 'status','imdb_id', 'logRevenue', 'logBudget',"released"]
with_duration = False
log_num_feats = ['budget', 'popularity']
cols_to_binarize = ['homepage','status']
cat_feats = ['has_homepage', 'release_month', 'release_year']
genre_feats = ['genre_Fantasy', 'genre_Action', 'genre_TV_Movie', 'genre_Romance',
               'genre_Western', 'genre_Animation', 'genre_Music', 'genre_Horror',
               'genre_History', 'genre_Mystery', 'genre_Family', 'genre_Drama',
               'genre_Science_Fiction', 'genre_War', 'genre_Adventure',
               'genre_Documentary', 'genre_Thriller', 'genre_Crime', 'genre_Comedy']
count_feats = ['production_countries_count', 'production_companies_count',
               'spoken_languages_count', 'keyword_count', 'Duration']

today = datetime(2024, 4, 14)


def change_name(df, old_name="keywords", new_name='Keywords'):
    if old_name in df.columns:
        df = df.rename(columns={old_name: new_name})
    return df


cols_to_drop = ["title","vote_average","vote_count","runtime","adult","backdrop_path","original_title","overview","tagline","original_title"]


def set_cols(df, cols_to_drop = cols_to_drop):
    all_columns = df.columns
    df = df[list(set(all_columns).difference(set(cols_to_drop)))]
    return df


def remove_negative_money(df):
    df_ = df[df.budget > 0].copy()
    return df_


def fillnan(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Remplacer les valeurs manquantes par le mode pour les colonnes catégorielles
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # Remplacer les valeurs manquantes par la médiane pour les colonnes numériques
            df[col].fillna(df[col].median(), inplace=True)
    return df





def Binarizer(df, cols_to_binarize):
    df_copy = df.copy()
    for col in cols_to_binarize:
        if col=='status':
            num_col_name ="released"
        elif col == "homepage":
            num_col_name = "has_homepage"

        df_copy[num_col_name] = 1
        df_copy.loc[pd.isnull(df_copy[col]), num_col_name] = 0
    df_copy = df_copy.drop(cols_to_binarize,axis=1).copy()
    return df_copy




def count_strings(s):
    if pd.isna(s):
        return np.nan
    return len(s.split(','))


def remove_empty_date_line(X_df, date_col=date_col):
    X_df = X_df[X_df[date_col].notnull()]
    return X_df

def yearfix(x):
    r = x[:4]
    return int(r)

def apply_yearfix(df, date_col=date_col, col_name="release_year"):
    df[col_name] = df[date_col].apply(lambda x: yearfix(x))
    return df

def monthfix(x):
    r = x[5:7]
    return int(r)

def apply_monthfix(df, date_col=date_col, col_name="release_month"):
    df[col_name] = df[date_col].apply(lambda x: monthfix(x))
    return df 

def str_to_datetime(str_date, today):
    date_reference = datetime.strptime(str_date, "%Y-%m-%d")
    difference = today - date_reference
    return round(difference.total_seconds() / (3600 * 24), 5)

def add_duration_col(df, with_duration=False, date_col="release_date"):
    if with_duration:
        # Convertir la colonne date_col en chaînes de caractères (str)
        df[date_col] = df[date_col].astype(str)
        today = datetime(2024, 4, 14)
        df["Duration"] = df[date_col].apply(lambda x: str_to_datetime(x, today))
    return df

def model_features(df, cols_to_remove=Cols_to_Remove):
    return list(set(df.columns) - set(cols_to_remove))

class Log1pTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, features=None):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)

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
    


def apply_count(df, cols_to_count_values):
    df_copy = df.copy()
    for col in cols_to_count_values:
        new_col_name = col + "_count"
        df_copy[new_col_name] = df_copy[col].apply(count_strings)
    df_final = df_copy.drop(cols_to_count_values, axis=1)
    return df_final


def add_gender_cols(df):
    X_all_genres = df['genres'].str.split(', ', expand=True)
    X_all_genres = pd.get_dummies(X_all_genres.apply(lambda x: pd.Series(x).str.strip()), prefix='', prefix_sep='').groupby(level=0, axis=1).sum()
    X_all_genres.columns = ['genre_' + col.replace(' ', '_') for col in X_all_genres.columns]
    df1 = df.drop(["genres"],axis=1).copy() 
    X_all = pd.concat([df1, X_all_genres], axis=1).copy()
    return X_all
    
def genre_column_names(all_cols):
    # Séquence à rechercher
      sequence = 'genre_'

# Filtrer les éléments de la liste qui commencent par la séquence
      genre_column_names = [element for element in all_cols if element.startswith(sequence)]
      return list(genre_column_names)

def preprocessing_pipeline(df,ct):
    # Appliquer les transformations sur df
    big_df = change_name(df).copy()
    big_df = set_cols(big_df)
    big_df = remove_negative_money(big_df)
    df_with_count_col = apply_count(big_df, cols_to_count_values=cols_to_count_values)
    big_df = Binarizer(df_with_count_col, cols_to_binarize)  # Si Binarizer est une fonction définie ailleurs
    big_df = add_gender_cols(big_df).copy()  # Si add_gender_cols est une fonction définie ailleurs
    big_df= remove_empty_date_line(big_df)  # remplacer le deuxieme argument par y et recuperer y_big
    big_df=fillnan(big_df)
    big_df = add_duration_col(big_df, with_duration=with_duration)
    big_df = apply_monthfix(big_df)
    big_df = apply_yearfix(big_df)
    big_df = big_df.drop(date_col, axis=1)
    genre_feats = genre_column_names(big_df.columns)  # Si genre_column_names est une fonction définie ailleurs

    # Définir les colonnes à utiliser dans le ColumnTransformer
    # cat_feats = ['has_homepage']
    count_feats = ['spoken_languages_count', 'production_countries_count',
       'production_companies_count', 'Keywords_count']
    if with_duration==True:
        num_feats = ["Duration"] # non requis, à explorer ultérieurement
        names = log_num_feats + cat_feats + genre_feats + count_feats + num_feats + ['has_homepage', 'released'] 
    else :
        names = log_num_feats + cat_feats + genre_feats + count_feats + ['has_homepage', 'released'] 

    X = big_df[names]  # Pas besoin de copier car vous l'avez déjà fait lors de la transformation
    X = X.loc[:, ~X.columns.duplicated()]
    transformed_df = ct.transform(X)

    

    return transformed_df
    






