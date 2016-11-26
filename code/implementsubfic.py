#!/usr/bin/env python3

# implementsubfic.py

# based on implementmodel.py, with alterations that streamline and
# generalize it for simpler problems. Basically, I take out the
# complicated ad-hoc rules I used to combine models in implementmodel,
# and simply report probabilities.

# This script still loads feature files, converts them to a dictionary of features
# normalized according to a predefined scheme, and then uses an ensemble of genre
# models to make predictions about genre.


import csv, os, sys, pickle, glob
from collections import Counter
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# import utils
currentdir = os.path.dirname(__file__)
libpath = os.path.join(currentdir, '../lib')
sys.path.append(libpath)

import SonicScrewdriver as utils
import parsefeaturejsons as parser

# Establish some global variables:

# Let's create a list of the top 1000 English words so we can check
# the probability that a book is written in English

dictionarypath = os.path.join(libpath, 'top1000words.txt')
top1000words = set()
with open(dictionarypath, encoding = 'utf-8') as f:
    for line in f:
        top1000words.add(line.strip())

def get_metadata(metapath):
    ''' Returns the metadata as a pandas dataframe, translating strings
    to simpler boolean flags.
    '''

    meta = pd.read_csv(metapath, index_col = 'docid', dtype = 'object')

    return meta

# We read the models as pickled files; each one contains a dictionary with
# the following keys:
#     name
#     positivelabel     genre the model identifies
#     negativelabel     genre(s) from which it is discriminated
#     ftcount           number of features
#     vocabulary        an actual list of the features
#     c                 c parameter used in fitting the svm
#     n                 the total number of pos & neg instances in training set
#     scaler            a StandardScaler object with means and std devs for features
#     svm               the model itself, an svm.SVC object


def loadamodel(modelpath):

    with open(modelpath, mode = 'rb') as f:
        modelbytes = f.read()
        model = pickle.loads(modelbytes)
    print(model['positivelabel'], model['negativelabel'])

    return model

def counts4file(filepath):
    '''
    Gets counts for a single file.
    Importantly, this is exactly the same
    script used in trainamodel.
    '''

    counts = Counter()
    try:
        with open(filepath, encoding = 'utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['feature']
                if len(word) < 1:
                    continue

                ct = float(row['count'])

                if word.startswith('#header'):
                    word = word.replace('#header', '')
                #
                # This debatable choice treats header words as equivalent
                # to occurrences in the body text. In practice, this seems
                # to slightly improve performance, at least when you're using
                # SVMs and relatively low numbers of features (140-300).
                # Otherwise header words are in practice just discarded, because
                # e.g. #headeract won't be one of the top 250 words.

                counts[word] += ct

        return counts, 'success', 0

    except:
        return counts, 'file not found', 0

def onevolume2frame(vocabulary, counts):
    '''
    This version of countdict2featureframe is designed to
    accept a dictionary of counts for a single file, and
    return a pandas dataframe with columns specified by
    the vocabulary.

    Again, it's important that this code should continue to
    match the version in trainamodel. In a refactoring, you would
    extract this, and counts4file, and put them in a library.
    '''

    df = dict()
    # We initially construct the data frame as a dictionary of Series.
    vocabset = set(vocabulary)

    for v in vocabulary:
        df[v] = pd.Series([0])
        if v in counts:
            df[v].iloc[0] = counts[v]

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df)
    df = df[vocabulary]
    # This reorders the columns to be in vocab order

    return df

def prediction_for_file(model, counts):
    vocabulary = model['vocabulary']
    df = onevolume2frame(vocabulary, counts)
    scaler = model['scaler']
    scaleddf = scaler.transform(df)
    supportvector = model['svm']

    # The [0][1] at the end of the next two lines select the
    # prediction for the first volume [0] and positive class [1].

    prediction = supportvector.predict(scaleddf)[0]
    probability = supportvector.predict_proba(scaleddf)[0][1]

    return prediction, probability

def make_genredict(metadata, docid):
    ''' This converts a row of a pandas dataframe into a
    simpler dictionary representation. We expect all of
    these keys to have boolean value.
    '''
    genres = dict()

    for g in ['bio', 'dra', 'fic', 'poe']:
        genres[g] = metadata.loc[docid, g]
        if genres[g] != True and genres[g] != False:
            print('Unexpected value in genredict.')

    return genres


def volume_classification(models, counts4volume):

    nonfictionprob = 0.5
    juvenileprob = 0.5

    for name, model in models.items():

        positiveclass = model['positivelabel']

        prediction, probability = prediction_for_file(model, counts4volume)

        if positiveclass == 'rin':
            nonfictionprob = probability
        elif positiveclass == 'juv':
            juvenileprob = probability
        else:
            print('Wtf. What kind of model are you giving me?')

    return nonfictionprob, juvenileprob

def get_english_percent(counts, top1000englishwords):
    '''
    Percent of words in a volume that are contained in the set
    top1000words; to be used later to weed out non-English books.
    '''
    allalpha = 0
    allenglish = 0

    for word, count in counts.items():
        if not word.isalpha():
            continue
        else:
            allalpha += count
            if word in top1000englishwords:
                allenglish += count

    if allalpha == 0:
        return 0
    else:
        return allenglish / allalpha


def get_pairtree(pairtreeroot, htid):

    path, postfix = utils.pairtreepath(htid, pairtreeroot)
    wholepath = path + postfix + '/' + postfix + '.json.bz2'

    return wholepath

def counts4json(path, docid):
    if os.path.isfile(path):
        try:
            volume = parser.VolumeFromJson(path, docid)
            counts, totalwords = volume.get_volume_features()
            flatcounts = Counter()
            for key, value in counts.items():
                if key.startswith('#header'):
                    newkey = key.replace('#header', '')
                    flatcounts[newkey] += value
                else:
                    flatcounts[key] += value

            return flatcounts, 'success', totalwords

        except:
            return Counter(), 'parsing failure', 0
    else:
        return Counter(), 'file not found', 0


def main(sourcedir, metapath, modeldir, outpath, pairtree = False):
    '''
    This function can be called from outside the module; it accepts
    path information and then iterates through all the files it
    finds in the metadata at "metapath."

    If the pairtree flag is True, we assume sourcedir is the root
    of a pairtree structure. Otherwise we assume it's a flat list.
    '''

    global allnames, top1000words

    alternatesource = '/projects/ichass/usesofscale/post23/englishmonographs1980-2016/'

    # We're going to store all the models, by name, in a dictionary:

    models = dict()

    modelpaths = glob.glob(modeldir + '*.p')

    for apath in modelpaths:
        name = apath.replace(modeldir, '')
        name = name.replace('.p', '')
        models[name] = loadamodel(apath)

    # Now get metadata.

    metadata = get_metadata(metapath)

    nonficprobs = []
    juvieprobs = []
    wordcounts = []

    c = 0
    for docid in metadata.index:
        print(c)
        c += 1

        if pairtree:
            path1 = get_pairtree(sourcedir, docid)
            path2 = get_pairtree(alternatesource, docid)

            if os.path.isfile(path1):
                chosenpath = path1
            elif os.path.isfile(path2):
                chosenpath = path2
            else:
                print(path1)
                print(path2)
                print('file not found')
                error = 'file not found'
                wordcount = 0

            counts, error, wordcount = counts4json(chosenpath, docid)

        else:
            path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.csv')
            counts, error, wordcount = counts4file(path)

        if error == 'success':
            nonficprob, juvenileprob = volume_classification(models, counts)
        else:
            nonficprob = 0.5
            juvenileprob = 0.5

        nonficprobs.append(nonficprob)
        juvieprobs.append(juvenileprob)
        wordcounts.append(wordcount)


    metadata.loc[ : , 'nonficprob'] = pd.Series(nonficprobs, index = metadata.index)
    metadata.loc[ : , 'juvenileprob'] = pd.Series(juvieprobs, index = metadata.index)
    metadata.loc[ : , 'wordcount'] = pd.Series(wordcounts, index = metadata.index)

    metadata.to_csv(outpath)

if __name__ == '__main__':

    args = sys.argv

    if len(args) == 3:
        sourcedir = '/projects/ichass/usesofscale/post23/englishmonographs1920-79/'
        metapath = args[1]
        modeldir = '../ficmodels/'
        outpath = args[2]
        main(sourcedir, metapath, modeldir, outpath, pairtree = True)

    else:
        main(sourcedir = '/Volumes/TARDIS/work/train20', metapath = 'maintrainingset.csv', modeldir = 'models/', outpath = 'predicted_metadata.csv', pairtree = False)




















