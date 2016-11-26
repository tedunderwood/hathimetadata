#!/usr/bin/env python3

# Based on

# trainamodel.py

# With edits as needed to make this work for page-level modeling.
# I am going to try to refactor so both processes can run on the
# same code, but have not done that yet. May be challenging!

# Given two categories, represented by positivelabel and negativelabel,
# these functions identify pages in the classes,
# select features by finding the n most common words in those pages,
# by page frequency, and then train
# models of the classes using those features.

# n is consistently number of features,
# c is a constant passed to the svm, which needs to be optimized,
# and k is the number of folds in cross-validation

# We expect the page metadata to contain columns "class" -- 0 or 1
# and "groupid" -- which will be the docid for the volume.

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
libpath = os.path.join(currentdir, '../../lib')
sys.path.append(libpath)

import SonicScrewdriver as utils
import parsefeaturejsons as parser

def get_page_metadata(metapath):
    ''' Returns the metadata as a pandas dataframe, and lists of positive and negative
    instance IDs.

    For page metadata, which is in some ways simpler than volume-level metadata, because
    it already has a 'class' column.
    '''
    meta = pd.read_csv(metapath, index_col = 'pageid', dtype = 'object')

    allpageIDs = meta.index.values.tolist()

    docids = meta['groupid'].unique()

    return meta, allpageIDs, docids

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

def get_vocabulary_and_counts_4pages(metadata, allIDs, sourcedir, n):
    ''' Gets the top n words by docfrequency, but also
    returns a dictionary of wordcounts so we don't have to read them again from the
    file when generating a feature dataframe.

    Adjusted to handle page instances.
    '''

    doc_freq = Counter()
    counts = dict()
    id2group = dict()

    for docid in allIDs:

        path = os.path.join(sourcedir, utils.clean_pairtree(docid) + '.basic.json.bz2')
        volume = parser.PagelistFromJson(path, docid)
        pagecounts = volume.get_feature_list()

        for idx, page in enumerate(pagecounts):
            pageid = docid + '||' + str(idx)

            id2group[pageid] = docid

            counts[pageid] = page
            for key, value in page.items():
                doc_freq[key] += 1

    vocab = [x[0] for x in doc_freq.most_common(n)]
    print('Vocabulary constructed.')

    return vocab, counts, id2group

def countdict2featureframe_4pages(vocabulary, allpageIDs, counts, id2group):
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

    groupids = []

    for pageid in allpageIDs:
        if pageid not in id2group:
            print('Error in constructing id2group.')
            print(pageid)
            continue
        else:
            groupids.append(id2group[pageid])
            for feature, count in counts[pageid].items():
                if feature in vocabset:
                    df[feature].loc[pageid] = count

    # Now let's refashion the dictionary as an actual dataframe.
    df = pd.DataFrame(df, index = allpageIDs)
    df = df[vocabulary]

    # This reorders the columns to be in vocab order

    df.loc[ : , 'groupid'] = pd.Series(groupids, index = allpageIDs)

    return df

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

def svm_model_one_fold(metadata, fold, grouped_df, c, featurecount):
    ''' This constitutes a training set by excluding volumes in fold,
    constitutes a training set that *is* fold, and then fits a linear
    SVM using parameter c. You can also specify a reduced subset of
    features, if desired; this becomes useful when we're tuning
    and trying to identify the size of an optimal feature set.
    '''

    data = grouped_df.loc[-grouped_df['groupid'].isin(fold), : ]
    data = data.drop('groupid', 1)

    testdata = grouped_df.loc[grouped_df['groupid'].isin(fold), : ]
    testdata = testdata.drop('groupid', 1)
    indices = testdata.index.tolist()

    # data is everything not in fold
    # testdata is everything that is in fold

    stdscaler = StandardScaler()
    stdscaler.fit(data)
    data = pd.DataFrame(stdscaler.transform(data), index = data.index)
    testdata = pd.DataFrame(stdscaler.transform(testdata), index = testdata.index)

    trainingyvals = metadata.loc[-metadata['groupid'].isin(fold), 'class']
    realyvals = metadata.loc[metadata['groupid'].isin(fold), 'class']

    # supportvector = svm.LinearSVC(C = c)
    supportvector = svm.SVC(C = c, kernel = 'linear', probability = True)
    supportvector.fit(data.loc[ : , 0: featurecount], trainingyvals)

    predictionpairs = supportvector.predict_proba(testdata.loc[ : , 0: featurecount])
    prediction = [float(x[1]) for x in predictionpairs]
    # this gets the prediction for the positive class, for each volume

    return prediction, realyvals, indices

def calculate_accuracy(prediction, realyvals):
    assert len(prediction) == len(realyvals)

    actualpositives = []
    predictedpositives = []
    actualnegatives = []
    predictednegatives = []
    for p, r in zip(prediction, realyvals):
        if float(p) > 0.5:
            predictedpositives.append(True)
            predictednegatives.append(False)
        else:
            predictedpositives.append(False)
            predictednegatives.append(True)

        if float(r) > 0.5:
            actualpositives.append(True)
            actualnegatives.append(False)
        else:
            actualpositives.append(False)
            actualnegatives.append(True)

    actualnegatives = np.array(actualnegatives)
    actualpositives = np.array(actualpositives)
    predictedpositives = np.array(predictedpositives)
    predictednegatives = np.array(predictednegatives)

    trueposct = sum(actualpositives & predictedpositives)
    truenegct = sum(actualnegatives & predictednegatives)
    falseposct = sum(actualnegatives & predictedpositives)
    falsenegct = sum(actualpositives & predictednegatives)

    assert len(realyvals) == trueposct + truenegct + falsenegct + falseposct

    accuracy = (trueposct + truenegct) / len(realyvals)
    precision = trueposct / (trueposct + falseposct)
    recall = trueposct / (trueposct + falsenegct)

    return accuracy, precision, recall

def cross_validate_svm_4pages(metadata, df, docids, allpageIDs, c, k, featurecount):
    ''' K-fold cross-validation of the model, using parameter c, and features
    up to featurecount.
    '''

    folds = []

    floor = 0
    for divisor in range(1, k + 1):
        pctceiling = divisor / k
        intceiling = int(pctceiling * len(docids))
        test = docids[floor : intceiling]
        folds.append(test)
        floor = intceiling

    allrealclasslabels = []
    allpredictions = []
    allindices = []
    # these have to be gathered because they will not be in original order

    for fold in folds:

        prediction, realclasslabels, indices = svm_model_one_fold(metadata, fold, df, c, featurecount)
        allrealclasslabels.extend(realclasslabels)
        allpredictions.extend(prediction)
        allindices.extend(indices)

    # The point of constructing allfolds is to reconstitute a list of IDs *in the same
    # order as* the predictions and real class labels, so it can be used to index them.

    allpredictions = pd.Series(allpredictions, index = allindices, dtype = 'float64')
    allrealclasslabels = pd.Series(allrealclasslabels, index = allindices, dtype = 'float64')

    accuracy, precision, recall = calculate_accuracy(allpredictions, allrealclasslabels)
    smoothedpredictions = split_trim_combine(allpredictions)
    smaccuracy, smprecision, smrecall = calculate_accuracy(smoothedpredictions, allrealclasslabels)

    return accuracy, precision, recall, smaccuracy, smoothedpredictions, allpredictions, allrealclasslabels

def tukeysmooth(inputseries):
    '''
    Not currently in use. Using probabilities turned out to be better.
    '''
    binsequence = list()
    for element in inputseries:
        if float(element) > 0.5:
            binsequence.append(1)
        else:
            binsequence.append(0)

    assert len(binsequence) == len(inputseries)

    newseq = list(binsequence)

    if len(binsequence) < 5:
        return newseq
    for i in range(1, len(binsequence) - 1):
        total = sum(binsequence[(i-1) : (i + 2)])
        # sums position i with the values before and after it
        if total < 2:
            newseq[i] = 0
        else:
            newseq[i] = 1

    return newseq

def split_trim_combine(aseries):
    pageindex = aseries.index.tolist()
    volindex = [x.split('||')[0] for x in pageindex]
    indexedbyvol = pd.Series(aseries.values, index = volindex)
    # note that it's quite important to say values there, so
    # you don't try to align indexes

    grouped = indexedbyvol.groupby(level = 0, sort = False)
    combined = []
    for agroup in grouped:
        name, thegroup = agroup
        trimmed = trimends(meansmooth(thegroup.values))
        # the strategy there is to smooth
        combined.extend(trimmed)

    assert len(combined) == len(aseries)

    return combined

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
    binsequence = list()

    for element in inputseries:
        if float(element) > 0.5:
            binsequence.append(1)
        else:
            binsequence.append(0)

    assert len(binsequence) == len(inputseries)
    if len(binsequence) < 5:
        return(binsequence)

    newseq = [1] * len(binsequence)
    newseq[0] = binsequence[0]
    newseq[-1] = binsequence[-1]

    for i in range(1, len(binsequence) - 1):
        total = sum(binsequence[(i-1) : (i + 2)])
        # sums position i with the values before and after it
        if total < 2:
            newseq[i] = 0
        else:
            newseq[i] = 1
            break

    for i in range(len(binsequence) - 2, -1, -1):
        total = sum(binsequence[(i-1) : (i + 2)])
        # sums position i with the values before and after it
        if total < 2:
            newseq[i] = 0
        else:
            newseq[i] = 1
            break

    return newseq


def model_gridtuple(gridtuple):
    ''' Cross-validates a single cell in a grid search. This is designed
    to be parallelized, so inputs and outputs are packed as tuples.
    '''

    vocab_subset, allpageIDs, docids, id2group, countdict, k, c, featurecount, metadata = gridtuple

    grouped_df = countdict2featureframe_4pages(vocab_subset, allpageIDs, countdict, id2group)
    accuracy, precision, recall, smaccuracy, smoothedpredictions, predictions, reallabels = cross_validate_svm_4pages(metadata, grouped_df, docids, allpageIDs, c, k, featurecount)
    result = k, c, featurecount, accuracy, precision, recall, smaccuracy, smoothedpredictions, predictions, reallabels

    print(featurecount, c, accuracy, smaccuracy)

    return result

def export_svm_model(metapath, sourcedir, ftcount, c, outpath4model):
    ''' Trains a model of positivelabel against negativelabel, where
    ftcount = number of features to use in the model (top n words)
    c = constant to be passed to the svm.

    Note: does no cross-validation.

    It then exports the model as a pickled file that contains the feature list,
    the scaler (means and standard deviations) used to normalize the data,
    and the model itself as an svm object, as well as metadata about parameter
    settings.
    '''

    print()
    print('Calculating a pagelevel model of ' + metapath)
    # print('Getting metadata ...')
    metadata, allpageIDs, docids = get_page_metadata(metapath)
    print('Loading wordcounts and getting features ...')
    vocabulary, countdict, id2group = get_vocabulary_and_counts_4pages(metadata, docids, sourcedir, ftcount)

    df = countdict2featureframe_4pages(vocabulary, allpageIDs, countdict, id2group)

    df = df.drop('groupid', 1)

    stdscaler = StandardScaler()
    stdscaler.fit(df)
    scaleddf = stdscaler.transform(df)
    scaleddf = pd.DataFrame(scaleddf, index = allpageIDs)

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
    model['ftcount'] = ftcount
    model['c'] = c
    model['n'] = len(allpageIDs)
    model['name'] = outpath4model.replace('.p', '')
    with open(outpath4model, mode = 'wb') as picklepath:
        pickle.dump(model, picklepath)

    print('Done. Model written to ' + outpath4model + '.')

    # # Lets now do probabilities, cross-validates

    # folds = break_into_folds(positiveIDs, negativeIDs, 5)

    # allrealclasslabels = []
    # allprobs = []
    # allfolds = []

    # for fold in folds:

    #     prediction, probabilities, realclasslabels = svm_probabilistic_one_fold(metadata, fold, scaleddf, c, ftcount)
    #     allrealclasslabels.extend(realclasslabels)
    #     allprobs.extend(probabilities)
    #     allfolds.extend(fold)

    # probapath = outpath4model.replace('.p', '.probs.csv')
    # with open(probapath, mode = 'w', encoding = 'utf-8') as f:
    #     writer = csv.writer(f)
    #     for idx, anID in enumerate(allIDs):
    #         row = [anID, allprobs[idx], allrealclasslabels[idx]]
    #         writer.writerow(row)

def export_logistic_model(metapath, sourcedir, ftcount, c, outpath4model):
    ''' Trains a model of positivelabel against negativelabel, where
    ftcount = number of features to use in the model (top n words)
    c = constant to be passed to the svm.

    Note: does no cross-validation.

    It then exports the model as a pickled file that contains the feature list,
    the scaler (means and standard deviations) used to normalize the data,
    and the model itself as an svm object, as well as metadata about parameter
    settings.
    '''

    print()
    print('Calculating a pagelevel model of ' + metapath)
    # print('Getting metadata ...')
    metadata, allpageIDs, docids = get_page_metadata(metapath)
    print('Loading wordcounts and getting features ...')
    vocabulary, countdict, id2group = get_vocabulary_and_counts_4pages(metadata, docids, sourcedir, ftcount)

    df = countdict2featureframe_4pages(vocabulary, allpageIDs, countdict, id2group)

    df = df.drop('groupid', 1)

    stdscaler = StandardScaler()
    stdscaler.fit(df)
    scaleddf = stdscaler.transform(df)
    scaleddf = pd.DataFrame(scaleddf, index = allpageIDs)

    yvals = metadata.loc[ : , 'class']

    # supportvector = svm.LinearSVC(C = c)
    logist = LogisticRegression(C = c)

    print('Training the model itself ...')
    logist.fit(scaleddf, yvals)

    prediction = logist.predict(scaleddf)
    accuracy, precision, recall = calculate_accuracy(prediction, yvals)

    print()
    print('Little trust should be placed in accuracy calculated on the sample')
    print('used to train the model, but for whatever it may be worth,')
    print('accuracy on this run was ' + str(accuracy))
    print()
    coefficients = logist.coef_[0] * 100

    coefficientuples = list(zip(coefficients, vocabulary))
    coefficientuples.sort()

    for coefficient, word in coefficientuples:
        print(word + " :  " + str(coefficient))
    print()
    print('Writing features and model to file ...')
    model = dict()
    model['vocabulary'] = vocabulary
    model['logistic'] = logist
    model['scaler'] = stdscaler
    model['ftcount'] = ftcount
    model['c'] = c
    model['n'] = len(allpageIDs)
    model['name'] = outpath4model.replace('.p', '')
    with open(outpath4model, mode = 'wb') as picklepath:
        pickle.dump(model, picklepath)

    print('Done. Model written to ' + outpath4model + '.')


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

    simple = True

    # export_svm_model('page/fic.csv', '/Users/tunder/work/pagedata', 100, 0.02, 'page/testmodel.p')

    if simple:
        # export_logistic_model('fic.csv', '/Users/tunder/work/pagedata', 100, 0.02, 'page/testmodel.p')
        export_svm_model('fic.csv', '/Users/tunder/work/pagedata', 86, 0.057, 'ficpagemodel.p')
    else:
        ftcount = 100
        k = 5
        c = 0.02
        metadata, allpageIDs, docids = get_page_metadata('page/fic.csv')
        print('Loading wordcounts and getting features ...')
        vocabulary, countdict, id2group = get_vocabulary_and_counts_4pages(metadata, docids, '/Users/tunder/work/pagedata', ftcount)
        gridtuple = vocabulary, allpageIDs, docids, id2group, countdict, k, c, ftcount, metadata

        result = model_gridtuple(gridtuple)
        k, c, featurecount, accuracy, precision, recall, smaccuracy, smoothedpredictions, predictions, reallabels = result












