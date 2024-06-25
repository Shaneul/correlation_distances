# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:55:57 2022

@author: shane
"""

import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir('C:\\Users\\shane\\Desktop\\CRT\\IndividualProject\\Scripts and Data')  # Provide the new path here
from Functions.CreateNetworks import Create_Network1, Create_Network2
from Functions.MLE_functions_edit import degree_list, fit, MLE, freqTable, empirical, PDF, CCDF, opt_single_dist
import csv
from scipy.special import factorial

Network_dict = {'Petster':['out.petster-hamster'], 'Munmun Digg' :['out.munmun_digg_reply'],
                  'Enron':['out.enron'], 'Human Proteins':['out.maayan-vidal'],
                  'Infectious Disease':['out.sociopatterns-infectious'],
                  'Astrophysics':['out.ca-AstroPh'],'Slashdot':['out.matrix'],
                  'Facebook':['out.facebook-wosn-links'],'US Powergrid':['out.opsahl-powergrid'],
                  'Moreno Health':['out.moreno_propro_propro'],
                  'Internet of Autonomous Systems 1':['out.as20000102'],
                  'Gene Fusion':['out.gene_fusion'],'PGP':['pgp.arenas-pgp'],
                  'Internet (CAIDA)':['out.as-caida20071105'],
                  'Java class dependencies':['out.subelj_jung-j_jung-j'],
                  'Internet of Autonomous Systems 2':['out.dimacs10-as-22july06'], 
                  'Gplus':['out.ego-gplus'], 'Twitter':['out.ego-twitter'],
                  'Facebook wall':['out.facebook-wosn-wall'],
                  'Linux':['out.linux'],'Reactome':['out.reactome'],
                  'Internet topology':['out.topology'] 
                  }
#Network_dict = {'Astrophysics':['out.ca-AstroPh']}
# Two lists: One for datasets with one header, one for two headers
Networks1h = ['out.petster-hamster','out.munmun_digg_reply','out.enron','out.gene_fusion','out.maayan-vidal', 'out.dimacs10-as-22july06','out.topology','out.reactome','out.linux']
Networks2h = ['out.sociopatterns-infectious','out.moreno_health_health','pgp.arenas-pgp','out.ca-AstroPh','out.ego-gplus','out.ego-twitter','out.matrix','out.facebook-wosn-links','out.facebook-wosn-wall','out.opsahl-powergrid','out.moreno_propro_propro','out.as20000102','out.as-caida20071105','out.subelj_jung-j_jung-j']


for i in list(Network_dict.keys()):
    if Network_dict[i][0] in Networks1h:
        Network_dict[i].append(1)
    else:
        Network_dict[i].append(2)
    
    
import pickle

myFile = open('data\\data_dictionary', 'wb')
myDict = Network_dict
print("The dictionary is:")
print(myDict)
pickle.dump(myDict,myFile)
myFile.close()
myFile = open('data\\data_dictionary', 'rb')
print("The content of the file after saving the dictionary is:")
print(pickle.load(myFile))    

G = Create_Network1('out.topology')
X = degree_list(G)    
result = MLE(X, 1, 3)

N, P, p = empirical(X)

ccdf = CCDF(result, X, N, P)

pdf = PDF(result, X, N, P)

plt.plot(N, ccdf)
plt.plot(N, P, '+', color='k')
plt.xscale('log')
plt.yscale('log')
plt.show()
pc10 = X[X>=50]


weightx = np.sort(np.append(X, pc10))

result_2 = MLE(weightx, 4, 3)

ccdf2 = CCDF(result_2, X, N, P)
plt.plot(N, ccdf2)
plt.plot(N, P, '+', color='k')
plt.xscale('log')
plt.yscale('log')
plt.show()

pdf2 = PDF(result, X, N, P)

plt.plot(N, pdf2)
plt.plot(N, p, '+', color='k')
plt.xscale('log')
plt.yscale('log')
plt.show()

from scipy.stats import lognorm
rvs = np.round(lognorm.rvs(result_2[2][0][1], size=len(X), scale=np.exp(result_2[2][0][0])))

_,_,rv_ln = freqTable(rvs)
_,_,dg = freqTable(X)
plt.scatter(list(rv_ln.keys()), list(rv_ln.values()), facecolors='none', edgecolors='blue')
plt.scatter(list(dg.keys()), list(dg.values()), facecolors='none', edgecolors='red')
plt.xscale('log')
plt.yscale('log')
plt.show()

from statsmodels.graphics.gofplots import qqplot_2samples

qqplot_2samples(X,rvs)

for i in [0,0.1,0.2,0.3]:
    cdf_test = CCDF(result_2, 7+i, N, P)
    print(cdf_test)
    
    
    
cdf = 1-ccdf2    
x10 = np.unique(X[X >= 10])
percentiles = []
for i in x10:
    percentile = cdf[np.where(N == i)]
    percentiles.append(percentile)
    
    
j = 0
pc_cdf = []
i = 0
while i < len(percentiles):    
    while 10+j < np.max(N):
        pc = 1 - CCDF(result_2, 10 + j, N, P)
        if np.abs((percentiles[i] - pc)) < 0.00005:
            pc_cdf.append(10+j)
        j+=.1
    i += 1    
j=0
while 10+j < np.max(N):
    pc = CCDF(result_2, 10 + j, N, P)
    print(pc)
    j+=.1
    
    
    