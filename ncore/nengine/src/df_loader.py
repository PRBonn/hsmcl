import pandas as pd


def load_pickle(picklePth):
    df = pd.read_pickle(picklePth)
    print(df.head(10))
    return df


def get_type(df, ind):
    row = df.iloc[ind]
    strg = row['type']
    return strg

def get_stamp(df, ind):
    row = df.iloc[ind]
    t = row['t']
    return t

def get_data(df, ind):
    row = df.iloc[ind]
    d = row['data']
    if type(d) is tuple:
        return d
    elif type(d) is str:
        return d
    elif type(d) is list:
        return d
    else:
        d = d.tolist()
    return d

def num_frames(df):
    num = len(df.index)
    return num