import pickle
import re, string
import pandas as pd
from numpy import savetxt, loadtxt
from sklearn.feature_extraction.text import TfidfVectorizer

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
def load_models(path):
    models = [None for _ in range(len(label_cols))]
    for i, filename in enumerate(label_cols):
        pickle_in = open(path + f"/{filename}.pickle","rb")
        models[i] = pickle.load(pickle_in)
        pickle_in.close()
    return models


re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): 
    return re_tok.sub(r' \1 ', s).split()


def load_vec(path):
    comment = pd.read_csv(path)["comment_text"]

    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
    trn_term_doc = vec.fit_transform(comment)

    return vec

def load_r(path):
    r = [None for _ in range(len(label_cols))]
    for i, filename in enumerate(label_cols):
        r[i] = loadtxt(path + f"/{filename}.csv", r[i])
    return r