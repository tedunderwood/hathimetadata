Metadata for English-Language Literature in HathiTrust Digital Library Beyond 1923
==================================================================================

This repository holds probabilistic inferred metadata pointing to English-language fiction (and, eventually poetry) in HathiTrust Digital Library. It covers volumes that were tagged as "in copyright," mostly dated after 1923, and is based on the v1.0 release of [the HathiTrust Research Center Extracted Features Dataset.](https://wiki.htrc.illinois.edu/display/COM/Extracted+Features+Dataset)

**If you want metadata before 1923,** that's already available as a finished product. You can access [the raw metadata, which points to particular volumes and pages that contain poetry, fiction, or drama](https://figshare.com/articles/Page_Level_Genre_Metadata_for_English_Language_Volumes_in_HathiTrust_1700_1922/1279201). (Then you'll have to pair the metadata with the actual text, or with wordcounts downloaded from the [HathiTrust Extracted Feature page](https://wiki.htrc.illinois.edu/display/COM/Extracted+Features+Dataset).) Or, for ease of use, you can directly download [word-frequency information for volumes in literary genres](https://wiki.htrc.illinois.edu/display/COM/Word+Frequencies+in+English-Language+Literature%2C+1700-1922).

This repository holds work in progress toward metadata for the period **beyond 1923.** Please note the phrase "work in progress." I haven't made this a stable data publication with a DOI, because the repo will keep changing and expanding throughout 2017, and perhaps 2018. (For instance, I'll add metadata for poetry, and quite possibly biography too.) But I wanted to share what I have, since it may be useful for other researchers.

how this was produced
---------------------
When I produced metadata before 1923, I wrote up [a full report detailing the process.](https://figshare.com/articles/Understanding_Genre_in_a_Collection_of_a_Million_Volumes_Interim_Report/12812) Part of the reason I'm calling this repo "work in progress" is that I don't have time to fully document the process yet. I've put some relevant scripts in the code folder, but they're not fully documented.

The process I've used here is slightly different from the process that produced the earlier metadata. Instead of training models that examine every book at the page level, I've moved toward a three-stage workflow:

1. Identify volumes of poetry, fiction, and biography, using a predictive model that works on word frequencies at the volume level.

2. Filter the corpus for certain predictable kinds of error (non-English-language volumes, juvenile fiction, etc).

3. Gather page-level training data *for each genre* and model pages *only within that genre.*

I think this multi-stage filtering process is more accurate than my earlier process, but exact figures for precision and recall are yet to come. (I can tell you precision and recall on the training and test sets, but that's not very meaningful. You really need to look at a separate sample of the collection — and since these are works in copyright, "looking at the works" is never easy.) For right now, I would just say, "here's some metadata if you want a way to start looking for fiction, but I can't provide warrantees about accuracy yet." 

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

If you use this metadata you can acknowledge it as Ted Underwood, "Metadata for English-Language Literature in HathiTrust Digital Library Beyond 1923," v0.1, Dec 2016, https://github.com/tedunderwood/hathimetadata




