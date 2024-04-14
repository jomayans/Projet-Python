import pandas as pd
import IPython.display as display

def load_data(in_path, name, n_display=1, show_info=False, sep=",", nrows=720000):
    df = pd.read_csv(in_path, sep=sep, nrows=nrows)
    print(f"{name}: shape is {df.shape}")
    if show_info:
        print(df.info())
    
    if n_display > 0:
        display(df.head(n_display))
    return df
