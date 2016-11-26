#!/usr/bin/env python3

# implementmodel.py

# This script loads feature files, converts them to a dictionary of features
# normalized according to a predefined scheme, and then uses an ensemble of genre
# models to make predictions about genre.

# The genre models are combined according to a fixed set of rules designed
# to maximize precision on the two genres we care most about: fiction and
# poetry.

# We begin by training three one-vs-all classifiers:
#   - fiction vs everything else
#   - poetry vs everything else
#   - drama vs everything else

# In order to improve recall, we are extremely generous with the cutoffs
# for flagging "fiction" and "poetry" at this stage, setting them
# at 10% for poetry and 20% for fiction.

# A volume that doesn't get flagged as fic or poe at all is of
# no further concern. If flagged (only) 'dra' or 'bio' we add it to
# low-precision lists of those volumes.

# If a volume *was* flagged 'fic' or 'poe,' we look for conflicting
# evidence. Conflicts can be created 1) by metadata predicting a
# different genre, or 2) by positive results on other one-vs-all
# classifiers, or 3) by a relatively low score on the classifier
# that did the flagging.

# To resolve the first two conflicts, we use one-vs-one models
# between the specific genres in conflict.
#    - fic versus poe
#    - fic versus dra
#    - poe versus dra
#    - fic versus bio
#    - poe versus bio

# To address the third concern, we force vols of fic and poe
# that got a low probability on one-vs-all (like, less than 75%)
# to also pass the gauntlet of a fic-vs-nonfiction or poe-vs-nonfiction
# classifier. (For this purpose nonfiction is folded in
# with biography.) The rationale here is that by far the majority
# of the collection is nonfiction, and these will be most of
# our errors. We want to be super-sure that our volumes don't
# look like nonfiction.

# Those are a lot of ad-hoc rules. But I made them up out of whole
# cloth, so ... I doubt they're "overfit" very tightly to
# the peculiarities of the training set.


import csv, os, sys, pickle
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

onevsallnames = ['ficvsall', 'poevsall', 'dravsall']
onevsonenames = ['ficvspoe', 'ficvsdra', 'ficvsbio', 'dravspoe', 'poevsbio', 'dravsbio']
gauntletnames = ['ficvsnonbio', 'poevsnonbio']
allnames = onevsallnames + onevsonenames + gauntletnames

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

    newdata = dict()
    newdata['bio'] = []
    newdata['fic'] = []
    newdata['poe'] = []
    newdata['dra'] = []

    for idx in meta.index:
        genres = str(meta.loc[idx, 'genres']).split('|')
        subjects = str(meta.loc[idx, 'subjects']).split('|')
        title = str(meta.loc[idx, 'title'].lower())

        dra = False
        fic = False
        poe = False
        bio = False

        if 'Biography' in genres or 'Biography' in subjects:
            bio = True

        if 'Autobiography' in genres or 'Autobiography' in subjects:
            bio = True

        if 'Description and travel' in subjects:
            bio = True

        if 'Fiction' in genres:
            fic = True

        if 'Poetry' in genres or 'poems' in title:
            poe = True

        if 'Drama' in genres or 'plays' in title:
            dra = True

        newdata['bio'].append(bio)
        newdata['dra'].append(dra)
        newdata['fic'].append(fic)
        newdata['poe'].append(poe)

    for genre in ['bio', 'dra', 'fic', 'poe']:
        meta.loc[ :, genre] = pd.Series(newdata[genre], index = meta.index)

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


def loadamodel(modelname):
    modelpath = modelname + '.p'        # because it's a pickle file
    with open(modelpath, mode = 'rb') as f:
        modelbytes = f.read()
        model = pickle.loads(modelbytes)
    print(modelname, model['positivelabel'], model['negativelabel'])

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


def volume_classification(models, counts4volume, genredict):
    global onevsallnames, onevsonenames, gauntletnames

    allgenres = ['bio', 'dra', 'fic', 'poe']

    # Basically, the purpose of onevsall classification is
    # to flesh out a genredict with preliminary information about
    # the *possibility* that this volume is in poe, dra, or fic.

    # The genredict will already have boolean values inferred from
    # metadata, some of which may already be True.

    onevsall_probs = dict()
    explanation = 'no competition'

    for m in onevsallnames:

        model = models[m]
        positiveclass = model['positivelabel']

        prediction, probability = prediction_for_file(model, counts4volume)

        if probability > 0.5:
            genredict[positiveclass] = True
        elif model == 'poevsall' and probability > 0.1:
            genredict[positiveclass] = True
        elif model == 'ficvsall' and probability > 0.25:
            genredict[positiveclass] = True

        onevsall_probs[positiveclass] = probability

    # Now we have a dict that could have True for any of four
    # genres: bio, dra, fic, and poe. We need to resolve conflicts.

    attested = []
    for genrekey, booleanvalue in genredict.items():
        if booleanvalue == True:
            attested.append(genrekey)

    if len(attested) < 1:
        return 'non', 1, explanation
        # by default, we assume nonfiction

    if len(attested) == 1:
        tocheck = attested[0]
        maxprob = 1
        explanation = 'one flagged'
        # we proceed to check the attested genre by running it
        # through necessary gauntlets

    else:
        # we have multiple attested genres and must resolve conflicts
        explanation = 'conflicts resolved'
        onevsone_probs = dict()
        for g in allgenres:
            onevsone_probs[g] = []

        for g1 in attested:
            for g2 in attested:
                if g1 == g2:
                    continue
                # We don't pit a genre against itself.

                # Now we need to find a model that pits g1 against g2, in
                # either order.

                firstorder = g1 + 'vs' + g2
                secondorder = g2 + 'vs' + g1

                if firstorder in models:
                    model2use = models[firstorder]
                elif secondorder in models:
                    model2use = models[secondorder]
                else:
                    # we have no model for this conflict
                    continue

                positiveclass = model2use['positivelabel']
                negativeclass = model2use['negativelabel']
                prediction, probability = prediction_for_file(model2use, counts4volume)

                onevsone_probs[positiveclass].append(probability)
                onevsone_probs[negativeclass].append(1 - probability)

        tocheck = 'non'
        maxprob = 0

        for g in allgenres:
            if len(onevsone_probs[g]) == 0:
                prob_of_g = 0
            else:
                prob_of_g = sum(onevsone_probs[g]) / len(onevsone_probs[g])

            if prob_of_g > maxprob:
                maxprob = prob_of_g
                tocheck = g

    if tocheck == 'non':
        return 'non', 1, '1v1 failure'
        print('Probable error: Not sure how we managed to get tocheck set to non')
        print('after other genres were attested.')

    elif tocheck == 'dra':
        return 'dra', (maxprob + onevsall_probs['dra']) / 2, explanation

    elif tocheck == 'bio':
        return 'bio', maxprob, explanation

    else:
        assert (tocheck == 'fic' or tocheck == 'poe')
        assert tocheck in onevsall_probs

        previousprob = onevsall_probs[tocheck]

        if previousprob < 0.70 and tocheck == 'poe' or previousprob < 0.85 and tocheck == 'fic':
            # we need to confirm this

            if tocheck == 'fic':
                confirmationmodel = models['ficvsnonbio']
            elif tocheck == 'poe':
                confirmationmodel = models['poevsnonbio']

            prediction, confirmation_probability = prediction_for_file(confirmationmodel, counts4volume)

            # this is now an up-or-down vote, based on the prediction,
            # but we do use the probability to calculate an overall
            # confidence score

            meanprobability = (previousprob + confirmation_probability) / 2

            if prediction > 0.5:
                return tocheck, meanprobability, explanation + ', confirmed'
            else:
                return 'non', 1 - meanprobability, explanation + ', confirmed'

        else:
            return tocheck, (onevsall_probs[tocheck] + 1) / 2, explanation + ', unconfirmed'

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

    # We're going to store all the models, by name, in a dictionary:

    models = dict()

    for name in allnames:
        models[name] = loadamodel(modeldir + name)

    # Now get metadata.

    metadata = get_metadata(metapath)

    predictedgenres = []
    predictedprobs = []
    explanations = []
    wordcounts = []
    englishpcts = []

    c = 0
    for docid in metadata.index:
        print(c)
        c += 1

        if pairtree:
            path = get_pairtree(sourcedir, docid)
            counts, error, wordcount = counts4json(path, docid)
        else:
            path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.csv')
            counts, error, wordcount = counts4file(path)

        if error == 'success':
            genredict = make_genredict(metadata, docid)
            englishpct = get_english_percent(counts, top1000words)
            genre, probability, explanation = volume_classification(models, counts, genredict)
        else:
            englishpct = 0
            genre = 'NA'
            probability = 0
            explanation = error

        predictedgenres.append(genre)
        predictedprobs.append(probability)
        explanations.append(explanation)
        wordcounts.append(wordcount)
        englishpcts.append(englishpct)

    metadata.loc[ : , 'predictedgenre'] = pd.Series(predictedgenres, index = metadata.index)
    metadata.loc[ : , 'probability'] = pd.Series(predictedprobs, index = metadata.index)
    metadata.loc[ : , 'wordcount'] = pd.Series(wordcounts, index = metadata.index)
    metadata.loc[ : , 'englishpct'] = pd.Series(englishpcts, index = metadata.index)
    metadata.loc[ : , 'explanation'] = pd.Series(explanations, index = metadata.index)

    metadata.to_csv(outpath)

if __name__ == '__main__':

    args = sys.argv

    if len(args) == 3:
        sourcedir = '/projects/ichass/usesofscale/post23/englishmonographs1920-79/'
        metapath = args[1]
        modeldir = '../models/'
        outpath = args[2]
        main(sourcedir, metapath, modeldir, outpath, pairtree = True)

    else:
        main(sourcedir = '/Volumes/TARDIS/work/train20', metapath = 'maintrainingset.csv', modeldir = 'models/', outpath = 'predicted_metadata.csv', pairtree = False)




















