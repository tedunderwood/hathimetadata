Metadata for English-Language Fiction beyond 1923
=================================================

This directory contains comma-separated tables with volume-level information, as well as a subdirectory of *page*-level information about those volumes. You only need the page-level information if you want to try trimming front and back matter. That may or may not be important, depending on your application.

ficmeta.csv
-----------

This is a list of volumes that I believe to contain fiction. It has not been manually checked; it was produced by predictive modeling, and reflects estimated probabilities. If you have high standards for accuracy, probably the best approach is to view this as a starting-place for further work. 

I have tried to err on the side of inclusiveness. This collection will include many more volumes of fiction than you would get if you relied purely on genre tags in the existing MARC records. But it will also include a certain number of volumes that are better characterized as biography, travel writing, or folklore. 

I have included some of the probabilities calculated by predictive models as columns in the database. This may be useful for researchers who want to define fiction less inclusively: a simple starting-point would be, to filter the dataset by demanding a higher threshold for inclusion on some of these columns.

Note also that this collection *has not been deduplicated.* When there are multiple copies or reprints of a work, it will occur multiple times in this table.

juvenileficmeta.csv
-------------------

Contains volumes that were flagged as "for a juvenile audience," either by manual metadata or by a predictive model.

key to columns in both metadata tables
--------------------------------------

A quick explanation of columns in the table. Note that I have often used pipe characters (|) as separators within a field.

**docid** HathiTrust Volume ID; this is a unique identifier for volumes.

**recordid** Identifies volumes that belong together.

**authordate** The author's dates of birth and death, when we have them.

**imprint** Place, publisher, and date are separated by (|) pipe characters. The date listed here may not always be the date provided in

**inferreddate** This is my best guess about date of publication. It is based mainly on the three columns **datetype, startdate,** and **enddate,** which are drawn from [character positions 06-14 of MARC field 008.](https://www.loc.gov/marc/bibliographic/bd008a.html) 

**place** This is a code, drawn from the MARC metadata, reflecting place of publication; to interpret it see the [MARC Code List for Countries.](https://www.loc.gov/marc/countries/)

**enumcron** When there is more than one volume in a work, this code disambiguates the volumes.

**subjects** and **geographics** are Library of Congress headings that a cataloger assigned to the work. Geographics indicate that the work is about a specific region.

**genres** are mostly Library of Congress genre/form headings, but this column has also been used to record some flags contained in [character position 33 of Marc field 008](https://www.loc.gov/marc/bibliographic/bd008b.html). You'll notice that in many cases this field states "Not fiction." That reflects the fact that a 0 (not fiction) was entered in this field. The unreliability of existing genre metadata is why I felt I needed to train predictive models.

**rawprobability** The initial probability of being fiction assigned by a model that was contrasting fiction to *everything else in HathiTrust*--i.e., poetry and drama as well as nonfiction

**englishtop1000pct** The fraction of the words in the book drawn from the top 1000 words in an English dictionary sorted by frequency. I used this to weed out works that weren't actually written in English, despite metadata saying they were.

**nonficprob** The probability that this work was nonfiction. The fiction/nonfiction boundary is tricky, so it was useful to train a model specifically for that boundary, leaving aside poetry, drama, etc.

**juvenileprob** I also trained a model to identify juvenile fiction. Volumes with a high probability in this column are set aside in a distinct table.

**metadatalikely** is a flag that is set to TRUE only if there was evidence in the **genres** field, or in the title, suggesting that this work was fiction

**metadatasuspicious** Indicates that there is evidence in the metadata militating against this being fiction. Only a very small number of these volumes are included, and several of those look like errors!

pages
-----
Folder containing page-level information about most of the volumes in **ficmeta** and about half of the volumes in **juvenileficmeta**.