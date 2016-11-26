Page-level data for fiction in HathiTrust, post-1923
====================================================

These are [JSON Lines](http://jsonlines.org) files, meaning that each line is a separate JSON object, for a different volume.

The files have been divided by date, using the **inferreddate** column of the metadata table.

Within each JSON object, you will find the following keys:

**docid** This is the HathiTrust Volume ID, the unique identifier for the volume.

**numpages** Total number of pages in the volume. This will also be the length of **predictions,** but I have broken it out separately for ease of use.

**predictions** A list of probabilities, one per page, estimating the probability that the page is fiction rather than front matter, back matter, or nonfiction. The sequence of pages is defined to be the same as the sequence in the 2016 release of the Extracted Feature files.

If you want to get every page that might be fiction, use this to select pages individually.

Alternatively, if you want to get a sequence of *contiguous* pages identified as fiction, you can use the two fields **firstpg** and **lastpg**, which will ignore isolated pages predicted to be fiction but floating improbably in a sea of nonfiction. This consolidation is done by the function **trimends** in the script **implementpagemodel.**

The important thing to know is that **firstpg** and **lastpg** are both *inclusive.* This is contrary to the way sequences are defined in many programming languages, where the "end" of a sequence actually lies outside the sequence.

data this points to
-------------------
NOTE that these probabilities were produced using the November, 2016 release of HTRC Extracted Features. **Page sequences can change when volumes are rescanned.** Right now there is unfortunately no way to trace a page across time, guaranteeing that page #P of volume V in 2019 will be the same page that was identified as page #P in 2016. So this metadata should ideally be paired with the Extracted Features release from late 2016. Otherwise there will be a small number of minor errors in the page mapping, because HathiTrust volumes do get rescanned, and the pagination does change.