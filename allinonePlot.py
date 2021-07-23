#!/usr/bin/env python3

import os
from hist import Hist
from matplotlib import pyplot as plt
import pickle as pkl
import mplhep as hep

import argparse
parser = argparse.ArgumentParser(description='Run quick plots')
parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")

opt = parser.parse_args()

def plot(h2016_nj,h2017_1j,h2017_2j):

    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(outdir+'_scaled'):
        os.makedirs(outdir+'_scaled')

    observables = h2016_nj.keys()
    for obs in observables:
      print(obs)
      if obs in ['wei','lep_eta','z_mass','jet_pt','dijet_pt']: continue 

      h_2016_nj = h2016_nj[obs]
      h_2017_1j = h2017_1j[obs]
      h_2017_2j = h2017_2j[obs]
      plt.gcf().clf()
      h_2016_nj.plot(label='2016 1+2j')
      h_2017_1j.plot(label='2017 1j')
      h_2017_2j.plot(label='2017 2j')
      plt.legend(prop={'size': 10})
      plt.gcf().savefig(f"{outdir}/{obs}.png")
      
      # print(type(h_2017_1j))
      h_2017_nj =  h_2017_1j +  h_2017_2j
      plt.gcf().clf()
      #fig, (ax1, ax2) = plt.subplots(nrows=2)
      hep.histplot(h_2016_nj, label='2016 1+2j')
      hep.histplot(h_2017_nj, label='2017 1+2j')
      #ax2.plot(h_2016_nj/h_2017_nj)
      h_2017_nj.plot_ratio(h_2016_nj)
      plt.legend(prop={'size': 10})
      #ax2.set_xlabel(obs)
      #ax2.set_ylabel('ratio')
      plt.gcf().savefig(f"{outdir}_scaled/{obs}.png")


def readFromPickles(inputfile):
    hists = pkl.load(open(inputfile,'rb'))

    return hists


if __name__ == "__main__":
    print("This is the __main__ part")
    
    h2016_nj = readFromPickles('plots_2016_nj_ZpT_250to400/Pickles.pkl')
    h2017_1j = readFromPickles('plots_2017_1j_ZpT_250to400/Pickles.pkl')
    h2017_2j = readFromPickles('plots_2017_2j_ZpT_250to400/Pickles.pkl')


    plot(h2016_nj,h2017_1j,h2017_2j)
