Metadata for English-Language Fiction beyond 1923.
=================================================

ficmeta.csv
-----------

This is a list of volumes that I believe to contain fiction. It has not been manually checked; it was produced by predictive modeling, and reflects estimated probabilities. If you have high standards for accuracy, probably the best approach is to view this as a starting-place for further work. 

I have tried to err on the side of inclusiveness. This collection will include many more volumes of fiction than you would get if you relied purely on genre tags in the existing MARC records. But it will also include a certain number of volumes that are better characterized as biography, travel writing, or folklore. 

I have included some of the probabilities calculated by predictive models as columns in the database. This may be useful for researchers who want to define fiction less inclusively: a simple starting-point would be, to filter the dataset by demanding a higher threshold for inclusion on some of these columns.

Note also that this collection *has not been deduplicated.* When there are multiple copies or reprints of a work, it will occur multiple times in this table.

**A quick explanation of columns in the table.**

**docid** HathiTrust Volume ID; this is a unique identifier for volumes.

**recordid** Identifies volumes that belong together.

**authordate** The author's dates of birth and death, when we have them.

**imprint** Place, publisher, and date are separated by (|) pipe characters. The date listed here may not always be the date provided in

**inferreddate** This is my best guess about date of publication. It is based on the four columns **datetype, startdate,** and **enddate,** which are drawn from [character positions 06-14 of MARC field 008.](https://www.loc.gov/marc/bibliographic/bd008a.html) For an explanation of thes columns see