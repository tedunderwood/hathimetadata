# filterframe.py

import csv, os, sys, random, glob
from collections import Counter
import numpy as np
import pandas as pd


allfic = pd.read_csv('annotatedfiction.csv', index_col = 'docid', dtype = 'object')

makenumeric = ['rawprobability', 'englishtop1000pct', 'nonficprob', 'juvenileprob']

for col in makenumeric:
    allfic.loc[ : , col] = pd.to_numeric(allfic.loc[ : , col], errors = 'coerce')

nonfic = allfic.loc[(allfic.metadatalikely != 'True') & (allfic.nonficprob > 0.9), : ]

allfic = allfic.loc[(allfic.metadatalikely == 'True') | (allfic.nonficprob <= 0.9), : ]

juvie = list()

juvie.extend(allfic.index[allfic.genres.str.contains('Juvenile audience')].tolist())
juvie.extend(allfic.index[allfic.genres.str.contains('Juvenile fiction')])
juvie.extend(allfic.index[allfic.genres.str.contains('Juvenile literature')])
juvie.extend(allfic.index[allfic.juvenileprob > 0.98])

juvie = list(set(juvie))

juvenile = allfic.loc[juvie, : ]

allfic = allfic.loc[~allfic.index.isin(juvie), : ]

biography = allfic.loc[(allfic.metadatasuspicious == 'True') & (allfic.metadatalikely != 'True'), : ]
allfic = allfic.loc[(allfic.metadatasuspicious != 'True') | (allfic.metadatalikely == 'True'), : ]

dubious = allfic.loc[(allfic.englishtop1000pct < 0.5) & (allfic.nonficprob > 0.5) & (allfic.metadatalikely != 'True'), : ]
allfic = allfic.loc[(allfic.englishtop1000pct > 0.5) | (allfic.nonficprob < 0.5) | (allfic.metadatalikely == 'True'), : ]

poems = allfic.loc[(allfic.metadatalikely != 'True') & (allfic.title.str.contains('poems')), : ]
allfic = allfic.loc[(allfic.metadatalikely == 'True') | (~allfic.title.str.contains('poems')), : ]

specialzaps = ['inu.32000007720735', 'mdp.39015010907569', 'mdp.39015063187572', 'mdp.39015063187572', 'uc1.32106017522977', 'uc1.b4430816', 'mdp.39015013800514', 'mdp.39076006905231', 'pst.000015802538', 'uc1.32106012856602', 'uc1.32106012856602', 'pst.000045129537', 'osu.32435050762517', 'wu.89101597433', 'mdp.39015041834386', 'mdp.39015048338365', 'uiug.30112039681736', 'mdp.39015021585230', 'mdp.39015021585230', 'mdp.39015066050181', 'mdp.39015014628948', 'uc1.32106002152475', 'pst.000005752386', 'pst.000005752386', 'uc1.32106000043411', 'mdp.39015013238111', 'mdp.39015013238111', 'uc1.b4967253', 'inu.30000060707829', 'inu.30000103178004', 'mdp.39015009850390', 'uiuo.ark:/13960/t2n59fd04', 'mdp.39015069174368', 'mdp.39015002741166', 'uc1.b3462058', 'uc1.32106002098025', 'mdp.39015069174368', 'inu.30000114424041', 'mdp.39015030299963', 'uiuo.ark:/13960/t7dr40h84']

with open('../crappyfiction.txt', encoding = 'utf-8') as f:
    crapfiction = [x.strip() for x in f.readlines()]

remainingcrap = [x for x in crapfiction if x in allfic.index]
specialzaps.extend(remainingcrap)
specialzaps = list(set(specialzaps))
zapped = allfic.loc[specialzaps, : ]
allfic = allfic.loc[~allfic.index.isin(specialzaps), : ]

nonfiction = pd.concat([nonfic, dubious])

nonfiction.to_csv('filterednonfiction.csv')
juvenile.to_csv('filteredjuvenile.csv')
biography.to_csv('filteredbiography.csv')
allfic.to_csv('filteredfiction.csv')
poems.to_csv('filteredpoems.csv')





