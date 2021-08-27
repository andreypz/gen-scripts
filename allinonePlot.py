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

lumi = 1 #fb
scales_150_250 = {
    "2016_nj":
    {
        "xs": 222.9,
        "fneg": 0.31,
        "Nev": 100000,
    },
    "2017_1j": 
    {
        "xs": 50.84,
        "fneg": 0.26,
        "Nev": 100000,
    },
    "2017_2j":
    {
        "xs": 49.88,
        "fneg": 0.38,
        "Nev": 100000,
    },
}

scales_250_400 = {
    "2016_nj":
    {
        "xs": 8.47,
        "fneg": 0.29,
        "Nev": 100000,
    },
    "2017_1j": 
    {
        "xs": 5.902,
        "fneg": 0.25,
        "Nev": 100000,
    },
    "2017_2j":
    {
        "xs": 5.655,
        "fneg": 0.37,
        "Nev": 100000,
    },
}

scales = scales_250_400


def plot(h2016_nj,h2017_1j,h2017_2j):

    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(outdir+'_scaled'):
        os.makedirs(outdir+'_scaled')

    observables = h2016_nj.keys()
    for obs in observables:
      print(obs)
      if obs in ['wei','lep_eta','z_mass','jet_pt','dijet_pt','nlep','njet','njet15']: continue 

      h_2016_nj = h2016_nj[obs]*scales["2016_nj"]["xs"]*1000*lumi/(1-2*scales["2016_nj"]["fneg"])/scales["2016_nj"]["Nev"]
      h_2017_1j = h2017_1j[obs]*scales["2017_1j"]["xs"]*1000*lumi/(1-2*scales["2017_1j"]["fneg"])/scales["2017_1j"]["Nev"]
      h_2017_2j = h2017_2j[obs]*scales["2017_2j"]["xs"]*1000*lumi/(1-2*scales["2017_2j"]["fneg"])/scales["2017_2j"]["Nev"]
      plt.gcf().clf()
      h_2016_nj.plot(label='2016 1+2j')
      h_2017_1j.plot(label='2017 1j')
      h_2017_2j.plot(label='2017 2j')
      plt.legend(prop={'size': 10})
      plt.gcf().savefig(f"{outdir}/{obs}.png")
      
      # print(type(h_2017_1j))
      h_2017_nj =  h_2017_1j +  h_2017_2j
      plt.gcf().clf()

      h_2017_nj.plot_ratio(h_2016_nj,     
                           rp_num_label="2017 1+2j",
                           rp_denom_label="2016 1+2j",     
                           rp_uncert_draw_type="line",
                           rp_uncertainty_type="poisson",
                           rp_ylim=[0.2, 2.2],
                       )

      plt.gcf().savefig(f"{outdir}_scaled/{obs}.png")


def readFromPickles(inputfile):
    hists = pkl.load(open(inputfile,'rb'))

    return hists


if __name__ == "__main__":
    print("This is the __main__ part")
    
    #h2016_nj = readFromPickles('plots_2016_nj_ZpT_260to390/Pickles.pkl')
    #h2017_1j = readFromPickles('plots_2017_1j_ZpT_260to390/Pickles.pkl')
    #h2017_2j = readFromPickles('plots_2017_2j_ZpT_260to390/Pickles.pkl')

    #h2016_nj = readFromPickles('plots_2016_nj_ZpT_160_240/Pickles.pkl')
    #h2017_1j = readFromPickles('plots_2017_1j_ZpT_160_240/Pickles.pkl')
    #h2017_2j = readFromPickles('plots_2017_2j_ZpT_160_240/Pickles.pkl')

    pickledCoffea = readFromPickles('plots_CoffeaHist/Pickles.pkl')
    print(pickledCoffea)

    plot(h2016_nj,h2017_1j,h2017_2j)
