# parallelize_model

# Script designed to parallelize trainamodel.py, especially for
# purposes of grid search to optimize parameters.

# This version has a couple of minor changes to make it work for page models

import csv
from multiprocessing import Pool
import matplotlib.pyplot as plt
import trainapagemodel as tmod
import numpy as np

def gridsearch(metapath, sourcedir, k, feature_start, feature_end, feature_inc, c_start, c_end, c_num, outpath):
    metadata, allpageIDs, docids = tmod.get_page_metadata(metapath)
    vocabulary, countdict, id2group = tmod.get_vocabulary_and_counts_4pages(metadata, docids, sourcedir, feature_end)

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
            gridtuple = (vocab_subset, allpageIDs, docids, id2group, countdict, k, c, featurecount, metadata)
            grid.append(gridtuple)

        ymade = True

    print(len(grid))
    print(yaxis)
    print(xaxis)

    pool = Pool(processes = 10)
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

            k, c, featurecount, accuracy, precision, recall, smaccuracy, smoothedpredictions, predictions, reallabels = result
            f1 = 2 * (precision * recall) / (precision + recall)
            writer.writerow([featurecount, c, accuracy, smaccuracy, precision, recall, f1])

            if featurecount in xaxis and c in yaxis:
                x = xaxis.index(featurecount)
                y = yaxis.index(c)
                matrix[x, y] = smaccuracy

    plt.rcParams["figure.figsize"] = [9.0, 6.0]
    plt.matshow(matrix, origin = 'lower', cmap = plt.cm.YlOrRd)

    coords = np.unravel_index(matrix.argmax(), matrix.shape)
    print(coords)
    print(xaxis[coords[0]], yaxis[coords[1]])

    plt.show()


if __name__ == '__main__':

    gridsearch('fic.csv', '/Users/tunder/work/pagedata', 5, 70, 120, 4, 0.043, 0.106, 10, 'pagegrid.csv')

