# parallelize_model

# Script designed to parallelize trainamodel.py, especially for
# purposes of grid search to optimize parameters.

import csv
from multiprocessing import Pool
import matplotlib.pyplot as plt
import trainamodel as tmod
import numpy as np

def gridsearch(metapath, sourcedir, positivelabel, negativelabel, k, feature_start, feature_end, feature_inc, c_start, c_end, c_num, outpath):
    metadata, positiveIDs, negativeIDs = tmod.get_metadata(metapath, positivelabel, negativelabel)
    vocabulary, countdict = tmod.get_vocabulary_and_counts(metadata, positiveIDs, negativeIDs, sourcedir, feature_end)

    print(vocabulary[0:100])

    grid = []

    xaxis = []
    ymade = False
    yaxis = []

    for featurecount in range(feature_start, feature_end, feature_inc):
        xaxis.append(featurecount)
        for c in np.linspace(c_start, c_end, c_num):
            if not ymade:
                yaxis.append(c)

            vocab_subset = vocabulary[0 : featurecount]
            gridtuple = (vocab_subset, positiveIDs, negativeIDs, countdict, k, c, featurecount, metadata)
            grid.append(gridtuple)

        ymade = True

    print(len(grid))
    print(yaxis)
    print(xaxis)

    pool = Pool(processes = 16)
    res = pool.map_async(tmod.model_gridtuple, grid)

    res.wait()
    resultlist = res.get()

    assert len(resultlist) == len(grid)

    pool.close()
    pool.join()

    xlen = len(xaxis)
    ylen = len(yaxis)
    matrix = np.zeros((xlen, ylen))

    with open(outpath, mode = 'a', encoding = 'utf-8') as f:
        writer = csv.writer(f)
        for result in resultlist:

            k, c, featurecount, accuracy, precision, recall, predictions, reallabels = result
            f1 = 2 * (precision * recall) / (precision + recall)
            writer.writerow([featurecount, c, accuracy, precision, recall, f1])

            if featurecount in xaxis and c in yaxis:
                x = xaxis.index(featurecount)
                y = yaxis.index(c)
                matrix[x, y] = accuracy

    plt.rcParams["figure.figsize"] = [9.0, 6.0]
    plt.matshow(matrix, origin = 'lower', cmap = plt.cm.YlOrRd)

    coords = np.unravel_index(matrix.argmax(), matrix.shape)
    print(coords)
    print(xaxis[coords[0]], yaxis[coords[1]])

    plt.show()


if __name__ == '__main__':

    gridsearch('/Volumes/TARDIS/work/german/germantraining.csv', '/Volumes/TARDIS/work/german/csvs/', 'nvl', 'non', 5, 100, 600, 50, 0.0001, .02, 8, '/Volumes/TARDIS/work/german/firstmodel.csv')

