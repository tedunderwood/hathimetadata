#!/usr/bin/env python3

import csv, os, sys, glob
from collections import Counter

# import utils
currentdir = os.path.dirname(__file__)
libpath = os.path.join(currentdir, '../lib')
sys.path.append(libpath)

import SonicScrewdriver as utils

filepaths = glob.glob('../outmeta/*.csv')
genrestoget = {'bio', 'dra', 'fic', 'poe'}

def write_genres(rows, firstflag):
    global genrestoget
    fieldnames = ['docid', 'recordid', 'oclc', 'locnum', 'author', 'authordate', 'imprint', 'inferreddate', 'datetype', 'startdate', 'enddate', 'imprintdate', 'place', 'enumcron', 'subjects', 'genres', 'geographics', 'contents', 'title', 'metadatalikely', 'metadatasuspicious', 'rawprobability', 'englishtop1000pct']

    for g in genrestoget:
        outpath = '../collatedmeta/' + g + 'post1922.csv'
        with open(outpath, mode = 'a', encoding = 'utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = fieldnames, extrasaction = 'ignore')
            if firstflag:
                writer.writeheader()
            for row in rows[g]:
                writer.writerow(row)


def initialize_rows():
    global genrestoget
    rows = dict()

    for g in genrestoget:
        rows[g] = []
    return rows

firstfile = True

for path in filepaths:
    rows = initialize_rows()
    print(path)

    # gather all the rows assigned to target genres
    # in this file

    with open(path, encoding = 'utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            pgenre = row['predictedgenre']
            if pgenre not in genrestoget:
                continue
                # that will get rid of a lot of 'non'
            english = float(row['englishpct'])
            if english < 0.21:
                continue
                # these volumes are an odd lot
                # around 0.22 you start getting
                # early english texts, so we'll
                # allow them in

            for g in genrestoget:
                if g != pgenre:
                    row.pop(g)
                    # There are columns in these rows for a Boolean flag
                    # that report whether there is any reason to identify
                    # this volume as bio, dra, fic, or poe. Pop the
                    # columns not related to our genre.
                else:
                    row['metadatalikely'] = row.pop(pgenre)

            row.pop('materialtype')
            row.pop('language')
            # we just don't need those; they're all "monograph - eng"

            genres = set(row['genres'].split('|'))
            subjects = set(row['subjects'].split('|'))
            title = set(row['title'].lower().split())

            if pgenre == 'fic' and ("Novel" in genres or "novel" in title):
                row['metadatalikely'] = True
            if pgenre == 'fic' and ("Fiction" in genres):
                row['metadatalikely'] = True
            if pgenre == 'poe' and ("poems" in title):
                row['metadatalikely'] = True

            if row['metadatalikely'] == True:
                likely = True
            else:
                likely = False
                row['metadatalikely'] = ''


            suspicious = False

            if pgenre != 'bio' and 'Directories' in subjects or 'Directories' in genres:
                suspicious = True
                # these are almost always errors;
                # personal names get
            if pgenre != 'bio' and 'Dictionary' in subjects or 'Dictionary' in genres:
                suspicious = True
                # these are almost always errors;
                # personal names get

            if pgenre != 'bio' and not likely and ("Biography" in genres or "Autobiography" in genres):
                suspicious = True
            if pgenre != 'bio' and  not likely and ("Biography" in subjects or "Autobiography" in subjects):
                suspicious = True
            elif pgenre != 'bio' and not likely and ("Description and travel" in genres or "Description and travel" in subjects):
                suspicious = True
            elif english < 0.45:
                suspicious = True

            if suspicious:
                row['metadatasuspicious'] = True
            else:
                row['metadatasuspicious'] = ''

            row['inferreddate'] = utils.date_row(row)
            row['rawprobability'] = row.pop('probability')
            row['englishtop1000pct'] = row.pop('englishpct')

            rows[pgenre].append(row)

    # now we have all the bio, dra, fic, and poe
    # write them to file

    write_genres(rows, firstfile)
    firstfile = False









