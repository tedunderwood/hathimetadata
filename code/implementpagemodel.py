#!/usr/bin/env python3

# implementpagemodel.py

# based on implementsubfic.py, with alterations that make it able to
# produce page-level predictions.

# This script still loads feature files, converts them to a dictionary of features
# normalized according to a predefined scheme, and then uses an ensemble of genre
# models to make predictions about genre.


import csv, os, sys, json, glob, pickle
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


def get_counts_4pages(path, docid):
    ''' Gets a dictionary of wordcounts.

    Adjusted to handle page instances.
    Same logic used in trainapagemodel, but
    simplified for one volume.
    '''

    volume = parser.PagelistFromJson(path, docid)
    pagecounts = volume.get_feature_list()
    error = 'success'
    # except:
    #     error = 'parsing error'
    #     return dict(), [], error

    counts = dict()
    pageids = []
    for idx, page in enumerate(pagecounts):
        pageid = docid + '||' + str(idx)

        pageids.append(pageid)

        counts[pageid] = page

    return counts, pageids, error

def loadamodel(modelpath):

    with open(modelpath, mode = 'rb') as f:
        modelbytes = f.read()
        model = pickle.loads(modelbytes)
    print('page model loaded')

    return model

def predict_volume(model, allpageIDs, counts, docid):
    ''' what it says on the label; returns a dictionary
    with page-level predictions for the volume; this will
    eventually be written out in json format
    '''
    vocabulary = model['vocabulary']
    df = pages2frame(vocabulary, allpageIDs, counts)
    pagepredictions = prediction_for_pages(model, df)
    firstpage, lastpage = trimends(meansmooth(pagepredictions))

    jsonobject = dict()
    jsonobject['docid'] = docid
    jsonobject['numpages'] = len(pagepredictions)
    jsonobject['pagepredictions'] = pagepredictions
    jsonobject['firstpage'] = firstpage
    jsonobject['lastpage'] = lastpage

    return jsonobject

def pages2frame(vocabulary, allpageIDs, counts):
    ''' Returns a pandas dataframe with feature counts for all the volumes
    to be used in this model. The dataframe is going to have an extra column
    that is used to group items for crossvalidation. E.g., if instances are pages,
    they might be grouped by volume ID for crossvalidating to avoid leaking info.
    If these are volumes, that might be the author ID.

    We expect positive and negative IDs to be the actual IDs of instances.

    Returns an unscaled data frame. Scaling is a separate step.
    '''

    df = dict()
    # We initially construct the data frame as a dictionary of Series.
    vocabset = set(vocabulary)

    for v in vocabulary:
        df[v] = pd.Series(np.zeros(len(allpageIDs)), index = allpageIDs)

    for pageid in allpageIDs:
        for feature, count in counts[pageid].items():
            if feature in vocabset:
                df[feature].loc[pageid] = count

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df, index = allpageIDs)
    df = df[vocabulary]

    # This reorders the columns to be in vocab order

    return df

def prediction_for_pages(model, df):
    scaler = model['scaler']
    scaleddf = scaler.transform(df)
    supportvector = model['svm']

    # The [0][1] at the end of the next two lines select the
    # prediction for the first volume [0] and positive class [1].

    probabilities = [x[1] for x in supportvector.predict_proba(scaleddf)]

    return probabilities

def meansmooth(inputseries):

    newseq = list(inputseries)

    if len(inputseries) < 5:
        return newseq
    for i in range(1, len(inputseries) - 1):
        newvalue = (inputseries[i] + (inputseries[i - 1] + inputseries[i + 1]) / 2) / 2
        # sums position i with the values before and after it, giving those
        # values half the weight of position i itself
        newseq[i] = newvalue

    return newseq

def trimends(inputseries):
    '''
    Returns the first page and last page considered
    to belong to the specified genre. Note that this
    is normal world "first" and "last," not the fracked-up
    programming-world definition of ranges where "last"
    is the place you stop, aka last+1.
    '''
    binsequence = list()

    for element in inputseries:
        if float(element) > 0.5:
            binsequence.append(1)
        else:
            binsequence.append(0)

    assert len(binsequence) == len(inputseries)
    if len(binsequence) < 5:
        return 0, len(binsequence) - 1

    newseq = [1] * len(binsequence)
    newseq[0] = binsequence[0]
    newseq[-1] = binsequence[-1]

    firstpage = 0
    lastpage = len(binsequence) - 1

    if binsequence[0] != 1:
        for i in range(1, len(binsequence) - 1):
            total = sum(binsequence[(i-1) : (i + 2)])
            # sums position i with the values before and after it
            if total < 2:
                newseq[i] = 0
            else:
                newseq[i] = 1
                firstpage = i
                break
    else:
        firstpage = 0
        # the first page was in-genre

    if binsequence[(len(binsequence) - 1)] != 1:
        for i in range(len(binsequence) - 2, -1, -1):
            total = sum(binsequence[(i-1) : (i + 2)])
            # sums position i with the values before and after it
            if total < 2:
                newseq[i] = 0
            else:
                newseq[i] = 1
                lastpage = i
                break
    else:
        lastpage = len(binsequence) - 1

    return firstpage, lastpage


def get_pairtree(pairtreeroot, htid):

    path, postfix = utils.pairtreepath(htid, pairtreeroot)
    wholepath = path + postfix + '/' + postfix + '.json.bz2'

    return wholepath


def main(sourcedirs, metapath, modeldir, outpath, pairtree = False):
    '''
    This function can be called from outside the module; it accepts
    path information and then iterates through all the files it
    finds in the metadata at "metapath."

    If the pairtree flag is True, we assume sourcedir is the root
    of a pairtree structure. Otherwise we assume it's a flat list.
    '''

    # We're going to store all the models, by name, in a dictionary:

    models = []

    modelpaths = glob.glob(modeldir + '*.p')
    assert len(modelpaths) == 1
    model = loadamodel(modelpaths[0])

    # Now get metadata.

    metadata = get_metadata(metapath)

    notfound = dict()

    c = 0
    path = ''

    for docid in metadata.index:
        print(c)
        c += 1

        if pairtree:
            found = False
            for sourcedir in sourcedirs:
                path = get_pairtree(sourcedir, docid)
                if os.path.isfile(path):
                    found = True
                    chosenpath = path
            if not found:
                print(path)
                print('file not found')
                error = 'file not found'
                wordcount = 0
            else:
                pagecounts, pageids, error = get_counts_4pages(chosenpath, docid)

        else:
            path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.csv')
            pagecounts, pageids, error = pagecounts4file(path)

        if error == 'success':
            volumejson = predict_volume(model, pageids, pagecounts, docid)
            volumestring = json.dumps(volumejson)
            with open(outpath, mode = 'a', encoding = 'utf-8') as f:
                f.write(volumestring + '\n')
            print(docid)
        else:
            notfound[docid] = error
            print(docid, error)


    with open('fictionpagesnotfound.txt', mode = 'a', encoding = 'utf-8') as f:
        for vol, reason in notfound.items():
            f.write(vol + '\t' + reason + '\n')

if __name__ == '__main__':

    args = sys.argv

    if len(args) == 3:
        sourcedirs = ['/projects/ichass/usesofscale/post23/englishmonographs1920-79/', '/projects/ichass/usesofscale/post23/englishmonographs1980-2016/']
        metapath = args[1]
        modeldir = '../pagemodels/'
        outpath = args[2]
        main(sourcedirs, metapath, modeldir, outpath, pairtree = True)

    else:
        main(sourcedir = '/Volumes/TARDIS/work/train20', metapath = 'maintrainingset.csv', modeldir = 'models/', outpath = 'predicted_metadata.csv', pairtree = False)




















