

from sklearn.model_selection import train_test_split  # sklearn.cross_validation in old versions

categorical = ['production_companies', 'production_countries', 'spoken_languages', 'Keywords']
cols_to_binerize = ['homepage', 'status', 'spoken_languages', 'Keywords']
cols_to_count_values = ['production_countries', 'production_companies', 'spoken_languages', 'Keywords', 'cast', 'crew']
# Formater la date pour obtenir le jour, le mois et l'année seulement
today = datetime(2024, 4, 14)
date_col="release_date"
Cols_to_Remove=['Keywords', 'spoken_languages','homepage', 'production_countries','production_companies', 'release_date', 'poster_path', 'id', 'status','imdb_id', 'logRevenue', 'logBudget',"released"]




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



from datetime import datetime

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


def pipeline(raw_df):
