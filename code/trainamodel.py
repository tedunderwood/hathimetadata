#!/usr/bin/env python3

# trainamodel.py

# Given two categories, represented by positivelabel and negativelabel,
# these functions identify volumes in the classes (get_metadata),
# select features by finding the n most common words in those volumes,
# by document frequency (get_vocabulary_and_counts), and then train
# models of the classes using those features.

# n is consistently number of features,
# c is a constant passed to the svm, which needs to be optimized,
# and k is the number of folds in cross-validation

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

def get_metadata(metapath, positivelabel, negativelabel):
    ''' Returns the metadata as a pandas dataframe, and lists of positive and negative
    instance IDs.
    '''
    meta = pd.read_csv(metapath, index_col = 'docid', dtype = 'object')

    positiveIDs = meta.loc[meta['volgenre'] == positivelabel, : ].index.tolist()

    # The negative label can come in three types. Either a straightforward label for
    # a specific class, or "everything but X," which takes the form of e.g. "~fic", or
    # two classes concatenated like 'nonbio'

    if len(negativelabel) > 5:
        labelA = negativelabel[0 : 3]
        labelB = negativelabel[3 : ]
        negativeAs = set(meta.loc[meta['volgenre'] == labelA, : ].index.tolist())
        negativeBs = set(meta.loc[meta['volgenre'] == labelB, : ].index.tolist())
        negativeIDs = list(negativeAs.union(negativeBs))
    elif not negativelabel.startswith('~'):
        negativeIDs = meta.loc[meta['volgenre'] == negativelabel, : ].index.tolist()
    elif negativelabel.startswith('~'):
        strippedlabel = negativelabel.replace('~', '')
        negativeIDs = meta.loc[meta['volgenre'] != strippedlabel, : ].index.tolist()

    print('We have ' + str(len(negativeIDs)) + ' negative and ')
    print(str(len(positiveIDs)) + 'positive IDs.')
    # Let's now limit metadata to volumes that are relevant to this model.
    allIDs = positiveIDs + negativeIDs
    meta = meta.loc[allIDs, : ]

    # and let's add a column to the metadata indicating whether these volumes
    # are in the positive or negative class
    classcolumn = pd.Series(np.zeros(len(allIDs)), index = allIDs)
    classcolumn.loc[positiveIDs] = 1
    meta.loc[ : , 'class'] = classcolumn

    return meta, positiveIDs, negativeIDs

def get_vocabulary(metadata, positiveIDs, negativeIDs, sourcedir, n):
    ''' Gets the top n words by docfrequency in positiveIDs + negativeIDs.
    '''

    allIDs = positiveIDs + negativeIDs

    doc_freq = Counter()

    for docid in allIDs:
        path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.csv')
        with open(path, encoding = 'utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:

                word = row['feature']
                if word.startswith('#header'):
                    word = word.replace('#header', '')

                doc_freq[word] += 1

    vocab = [x[0] for x in doc_freq.most_common(n)]
    print('Vocabulary constructed.')

    return vocab

def get_vocabulary_and_counts(metadata, positiveIDs, negativeIDs, sourcedir, n):
    ''' Gets the top n words by docfrequency in positiveIDs + negativeIDs, but also
    returns a dictionary of wordcounts so we don't have to read them again from the
    file when generating a feature dataframe.
    '''

    allIDs = positiveIDs + negativeIDs

    doc_freq = Counter()
    counts = dict()

    for docid in allIDs:
        counts[docid] = Counter()
        path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.csv')
        with open(path, encoding = 'utf-8') as f:
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

                doc_freq[word] += 1
                counts[docid][word] += ct

                # # experimental
                # if word.startswith('#'):
                #     squaredfeature = word + 'sqrd'
                #     counts[docid][word] = ct * ct


    vocab = [x[0] for x in doc_freq.most_common(n)]
    print('Vocabulary constructed.')

    return vocab, counts

def get_featureframe(vocabulary, positiveIDs, negativeIDs, sourcedir):
    ''' Returns a pandas dataframe with feature counts for all the volumes
    to be used in this model.
    '''

    df = dict()
    # We initially construct the data frame as a dictionary of Series.
    vocabset = set(vocabulary)
    allIDs = positiveIDs + negativeIDs

    for v in vocabulary:
        df[v] = pd.Series(np.zeros(len(allIDs)), index = allIDs)

    for docid in allIDs:
        path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.csv')
        with open(path, encoding = 'utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                feature = row['feature']

                if feature.startswith('#header'):
                    feature = feature.replace('#header', '')

                if feature in vocabset:
                    df[feature].loc[docid] = row['count']

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df, index = allIDs)
    df = df[vocabulary]
    # This reorders the columns to be in vocab order

    stdscaler = StandardScaler()
    scaleddf = pd.DataFrame(stdscaler.fit_transform(df), index = allIDs)

    return scaleddf

def countdict2featureframe(vocabulary, positiveIDs, negativeIDs, counts):
    ''' Returns a pandas dataframe with feature counts for all the volumes
    to be used in this model. Is a version of the previous function, except that
    instead of reading from file it uses in-memory.

    Returns a scaled frame.
    '''

    df = dict()
    # We initially construct the data frame as a dictionary of Series.
    vocabset = set(vocabulary)
    allIDs = positiveIDs + negativeIDs

    for v in vocabulary:
        df[v] = pd.Series(np.zeros(len(allIDs)), index = allIDs)

    for docid in allIDs:
        for feature, count in counts[docid].items():
            if feature in vocabset:
                df[feature].loc[docid] = count

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df, index = allIDs)
    df = df[vocabulary]
    # This reorders the columns to be in vocab order

    stdscaler = StandardScaler()
    scaleddf = pd.DataFrame(stdscaler.fit_transform(df), index = allIDs)

    return scaleddf

# def counts2frame_notscaled(vocabulary, positiveIDs, negativeIDs, counts):
#     ''' Returns a pandas dataframe with feature counts for all the volumes
#     to be used in this model.

#     Returns a frame that is not yet scaled, so that the scaler can be
#     built elsewhere. This function tends to be called when we're Building
#     a model to export, and need to export the scaler itself.
#     '''

#     df = dict()
#     # We initially construct the data frame as a dictionary of Series.
#     vocabset = set(vocabulary)
#     allIDs = positiveIDs + negativeIDs

#     for v in vocabulary:
#         df[v] = pd.Series(np.zeros(len(allIDs)), index = allIDs)

#     for docid in allIDs:
#         for feature, count in counts[docid].items():
#             if feature in vocabset:
#                 df[feature].loc[docid] = count

#     # Now let's refashion the dictionary as an actual dataframe.
#     df = pd.DataFrame(df, index = allIDs)
#     df = df[vocabulary]
#     # This reorders the columns to be in vocab order

#     return df

def counts2frame_notscaled(vocabulary, positiveIDs, negativeIDs, counts):
    ''' Returns a pandas dataframe with feature counts for all the volumes
    to be used in this model.

    Returns a frame that is not yet scaled, so that the scaler can be
    built elsewhere. This function tends to be called when we're Building
    a model to export, and need to export the scaler itself.
    '''

    df = dict()
    # We initially construct the data frame as a dictionary of Series.
    vocabset = set(vocabulary)
    allIDs = positiveIDs + negativeIDs

    for v in vocabulary:
        df[v] = pd.Series(np.zeros(len(allIDs)), index = allIDs)
        for docid in allIDs:
            if v in counts[docid]:
                df[v].loc[docid] = counts[docid][v]

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df, index = allIDs)
    df = df[vocabulary]
    # This reorders the columns to be in vocab order

    return df

def break_into_folds(positive, negative, k):
    ''' Breaks the positive and negative sets into k stratified
    folds containing equal numbers of pos and neg instances. These
    are embodied as lists of IDs.
    '''

    folds = []

    posfloor = 0
    negfloor = 0
    for divisor in range(1, k + 1):
        pctceiling = divisor / k
        intposceiling = int(pctceiling * len(positive))
        intnegceiling = int(pctceiling * len(negative))
        postest = positive[posfloor : intposceiling]
        negtest = negative[negfloor : intnegceiling]
        test = postest + negtest
        folds.append(test)
        posfloor = intposceiling
        negfloor = intnegceiling

    return folds

def svm_model_one_fold(metadata, fold, scaleddf, c, featurecount):
    ''' This constitutes a training set by excluding volumes in fold,
    constitutes a training set that *is* fold, and then fits a linear
    SVM using parameter c. You can also specify a reduced subset of
    features, if desired; this becomes useful when we're tuning
    and trying to identify the size of an optimal feature set.
    '''

    data = scaleddf.drop(fold)
    testdata = scaleddf.loc[fold, : ]
    trainingyvals = metadata.loc[~metadata.index.isin(fold), 'class']
    realyvals = metadata.loc[fold, 'class']

    # supportvector = svm.LinearSVC(C = c)
    supportvector = svm.SVC(C = c, kernel = 'linear', probability = True)
    supportvector.fit(data.loc[ : , 0: featurecount], trainingyvals)

    prediction = supportvector.predict(testdata.loc[ : , 0: featurecount])

    return prediction, realyvals

def svm_probabilistic_one_fold(metadata, fold, scaleddf, c, featurecount):
    ''' This constitutes a training set by excluding volumes in fold,
    constitutes a training set that *is* fold, and then fits a linear
    SVM using parameter c. You can also specify a reduced subset of
    features, if desired; this becomes useful when we're tuning
    and trying to identify the size of an optimal feature set.

    It differs from the preceding function in returning probabilities.
    '''

    data = scaleddf.drop(fold)
    testdata = scaleddf.loc[fold, : ]
    trainingyvals = metadata.loc[~metadata.index.isin(fold), 'class']
    realyvals = metadata.loc[fold, 'class']

    supportvector = svm.SVC(C = c, kernel = 'linear', probability = True)
    supportvector.fit(data.loc[ : , 0: featurecount], trainingyvals)

    predictions = supportvector.predict(testdata.loc[ : , 0: featurecount])
    probabilities = [x[1] for x in supportvector.predict_proba(testdata.loc[ : , 0: featurecount])]

    return predictions, probabilities, realyvals

def calculate_accuracy(prediction, realyvals):
    assert len(prediction) == len(realyvals)

    actualpositives = realyvals == 1
    predictedpositives = prediction == 1
    actualnegatives = realyvals != 1
    predictednegatives = prediction != 1

    trueposct = sum(actualpositives & predictedpositives)
    truenegct = sum(actualnegatives & predictednegatives)
    falseposct = sum(actualnegatives & predictedpositives)
    falsenegct = sum(actualpositives & predictednegatives)

    assert len(realyvals) == trueposct + truenegct + falsenegct + falseposct

    accuracy = (trueposct + truenegct) / len(realyvals)
    precision = trueposct / (trueposct + falseposct)
    recall = trueposct / (trueposct + falsenegct)

    return accuracy, precision, recall

def cross_validate_svm(metadata, positiveIDs, negativeIDs, scaleddf, c, k, featurecount):
    ''' K-fold cross-validation of the model, using parameter c, and features
    up to featurecount.
    '''

    folds = break_into_folds(positiveIDs, negativeIDs, k)

    allrealclasslabels = []
    allpredictions = []
    allfolds = []

    for fold in folds:

        prediction, realclasslabels = svm_model_one_fold(metadata, fold, scaleddf, c, featurecount)
        allrealclasslabels.extend(realclasslabels)
        allpredictions.extend(prediction)
        allfolds.extend(fold)

    # The point of constructing allfolds is to reconstitute a list of IDs *in the same
    # order as* the predictions and real class labels, so it can be used to index them.

    allpredictions = pd.Series(allpredictions, index = allfolds)
    allrealclasslabels = pd.Series(allrealclasslabels, index = allfolds)

    accuracy, precision, recall = calculate_accuracy(allpredictions, allrealclasslabels)

    return accuracy, precision, recall, allpredictions, allrealclasslabels

def basic_cross_validation(metapath, sourcedir, positivelabel, negativelabel, n, k, c):
    ''' Trains a model of positivelabel against negativelabel, where
    n = number of features to use in the model (top n words)
    c = constant to be passed to the svm, and
    k = number of folds in the cross-validation.
    '''

    metadata, positiveIDs, negativeIDs = get_metadata(metapath, positivelabel, negativelabel)
    vocabulary = get_vocabulary(metadata, positiveIDs, negativeIDs, sourcedir, n)
    scaleddf = get_featureframe(vocabulary, positiveIDs, negativeIDs, sourcedir)
    accuracy, precision, recall, predictions, reallabels = cross_validate_svm(metadata, positiveIDs, negativeIDs, scaleddf, c, k, n)

    print()
    print("Modeling " + positivelabel + " against " + negativelabel)
    print(accuracy, precision, recall)

    return predictions, reallabels

def model_gridtuple(gridtuple):
    ''' Cross-validates a single cell in a grid search. This is designed
    to be parallelized, so inputs and outputs are packed as tuples.
    '''

    vocab_subset, positiveIDs, negativeIDs, countdict, k, c, featurecount, metadata = gridtuple

    print(featurecount, c)

    scaleddf = countdict2featureframe(vocab_subset, positiveIDs, negativeIDs, countdict)
    accuracy, precision, recall, predictions, reallabels = cross_validate_svm(metadata, positiveIDs, negativeIDs, scaleddf, c, k, featurecount)
    result = k, c, featurecount, accuracy, precision, recall, predictions, reallabels

    return result

def export_svm_model(metapath, sourcedir, positivelabel, negativelabel, ftcount, c, outpath4model):
    ''' Trains a model of positivelabel against negativelabel, where
    ftcount = number of features to use in the model (top n words)
    c = constant to be passed to the svm.

    Note: does no cross-validation.

    It then exports the model as a pickled file that contains the feature list,
    the scaler (means and standard deviations) used to normalize the data,
    and the model itself as an svm object, as well as metadata about parameter
    settings.

    I assume that outpath4model is going to be a string ending in '.p' — i.e., a name for
    the pickle file. Without the .p, it becomes a name for the model; with ".probs.csv" added,
    it becomes a name for predicted, cross-validated probabilities.
    '''

    print()
    print('Calculating a model of ' + positivelabel + ' versus ' + negativelabel)
    # print('Getting metadata ...')
    metadata, positiveIDs, negativeIDs = get_metadata(metapath, positivelabel, negativelabel)
    print('Loading wordcounts and getting features ...')
    vocabulary, countdict = get_vocabulary_and_counts(metadata, positiveIDs, negativeIDs, sourcedir, ftcount)
    print('Building the data frame ...')
    df = counts2frame_notscaled(vocabulary, positiveIDs, negativeIDs, countdict)
    allIDs = df.index.tolist()
    stdscaler = StandardScaler()
    stdscaler.fit(df)
    scaleddf = stdscaler.transform(df)
    scaleddf = pd.DataFrame(scaleddf, index = allIDs)

    yvals = metadata.loc[ : , 'class']

    # supportvector = svm.LinearSVC(C = c)
    supportvector = svm.SVC(C = c, kernel = 'linear', probability = True)

    print('Training the model itself ...')
    supportvector.fit(scaleddf, yvals)

    prediction = supportvector.predict(scaleddf)
    accuracy, precision, recall = calculate_accuracy(prediction, yvals)

    print()
    print('Little trust should be placed in accuracy calculated on the sample')
    print('used to train the model, but for whatever it may be worth,')
    print('accuracy on this run was ' + str(accuracy))
    print()
    print('Writing features and model to file ...')
    model = dict()
    model['vocabulary'] = vocabulary
    model['svm'] = supportvector
    model['scaler'] = stdscaler
    model['positivelabel'] = positivelabel
    model['negativelabel'] = negativelabel
    model['ftcount'] = ftcount
    model['c'] = c
    model['n'] = len(allIDs)
    model['name'] = outpath4model.replace('.p', '')
    with open(outpath4model, mode = 'wb') as picklepath:
        pickle.dump(model, picklepath)

    print('Done. Model written to ' + outpath4model + '.')

    # Lets now do probabilities, cross-validates

    folds = break_into_folds(positiveIDs, negativeIDs, 5)

    allrealclasslabels = []
    allprobs = []
    allfolds = []

    for fold in folds:

        prediction, probabilities, realclasslabels = svm_probabilistic_one_fold(metadata, fold, scaleddf, c, ftcount)
        allrealclasslabels.extend(realclasslabels)
        allprobs.extend(probabilities)
        allfolds.extend(fold)

    probapath = outpath4model.replace('.p', '.probs.csv')
    with open(probapath, mode = 'w', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        for idx, anID in enumerate(allfolds):
            row = [anID, allprobs[idx], allrealclasslabels[idx], metadata.loc[anID, 'author'], metadata.loc[anID, 'title']]
            writer.writerow(row)


def onevolume2frame(vocabulary, counts):
    '''
    This version of countdict2featureframe is designed to
    accept a dictionary of counts for a single file, and
    return a pandas dataframe with columns specified by
    the vocabulary.
    '''

    df = dict()
    # We initially construct the data frame as a dictionary of Series.
    vocabset = set(vocabulary)

    for v in vocabulary:
        df[v] = pd.Series([0])

    for feature, ct in counts.items():
        if feature in vocabset:
            df[feature].iloc[0] = ct

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df)
    df = df[vocabulary]
    # This reorders the columns to be in vocab order

    return df

def apply_pickled_model(modeldict, counts):
    vocab = modeldict['vocabulary']
    scaler = modeldict['scaler']
    model = modeldict['svmobject']

    df = onevolume2frame(vocab, counts)
    scaleddf = scaler.transform(df)
    prediction = model.predict(scaleddf)

    return prediction[1]        # because that's the positive class

def counts4file(filepath):
    '''
    Gets counts for a single file.
    '''

    counts = dict()
    Counter()
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

    return counts

if __name__ == '__main__':

    export = True

    if export == True:

        export_svm_model('/Volumes/TARDIS/work/german/germantraining.csv', '/Volumes/TARDIS/work/german/csvs/', 'nvl', 'non', 500, 0.0035, '/Volumes/TARDIS/work/german/out/usethismodel.p')


    else:

        ficpredicts, ficlabels = basic_cross_validation('maintrainingset.csv', '/Volumes/TARDIS/work/train20', 'dra', '~dra', 800, 5, 0.02)

        meta = pd.read_csv('maintrainingset.csv', index_col = 'docid', dtype = 'object')

        classpredicts = dict()
        classestocheck = ['bio', 'poe', 'dra']

        classpredicts['bio'], biolabels = basic_cross_validation('maintrainingset.csv', '/Volumes/TARDIS/work/train20', 'bio', 'fic', 800, 5, 0.2)

        classpredicts['poe'], poelabels = basic_cross_validation('maintrainingset.csv', '/Volumes/TARDIS/work/train20', 'poe', 'fic', 800, 5, 0.2)

        classpredicts['dra'], dralabels = basic_cross_validation('maintrainingset.csv', '/Volumes/TARDIS/work/train20', 'dra', 'fic', 800, 5, 0.2)

        toflip = []
        for idx, value in ficpredicts.iteritems():
            for g in classestocheck:
                if value > 0.5 and meta.loc[idx, 'sampledas'] == g:
                    if idx in classpredicts[g] and classpredicts[g].loc[idx] > 0.5:
                        toflip.append(idx)
                    elif idx not in classpredicts[g]:
                        print(idx)

        ficpredicts.loc[toflip] = 0

        accuracy, precision, recall = calculate_accuracy(ficpredicts, ficlabels)
        print()
        print('After checking volumes where metadata encouraged us to doubt: ')
        print(accuracy, precision, recall)









