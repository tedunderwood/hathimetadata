Code used to create hathimetadata
=================================

This is not completely documented, but here are some notes.

trainamodel.py
--------------
Functions centrally needed for creating models of one genre against another genre, or of one genre against all other genres. Is called by ...

parallelmodel.py
----------------
Which uses multiprocessing to accelerate grid search, among other things.

implementmodel.py
-----------------
is the business end of this whole workflow, taking an ensemble of models produced by trainamodel.py and coordinating them (with some ad-hoc rules) in order to actually classify texts.

collate_results.py
------------------
Once the models have spit out predictions for several million volumes (mostly nonfiction); this script sorts through them and filters out the smaller subsets that look like drama, fiction, poetry, or biography.

trainapagemodel.py
---------------
Makes a page-level model.

parallelizepagemodel.py
-----------------------
Parallelizes the previous script so we can grid search on parameters.

implementpagemodel.py
---------------------
Implements the page-level model.

parsefeaturejsons.py
--------------------
Reads JSON files from HTRC extracted features. Called by the previous script.

implementsubfic.py
------------------
Trains models specifically of nonfiction, and juvenile fiction, mistakenly-included in the intermediate stage of ficmeta.csv,

filterframes.py 
---------------
Applies some adhoc rules to various probabilities produced by various models in order to exclude some stuff. **Note:** if you're looking for a place to be suspicious of my procedures, this ad hoc step is a good place.