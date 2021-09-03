#!/usr/bin/env python3

import os
#from hist import Hist
from coffea import hist
from matplotlib import pyplot as plt
import pickle as pkl
import mplhep as hep

import argparse
parser = argparse.ArgumentParser(description='Run quick plots')
parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
parser.add_argument('-t','--inputType', type=int, default=0, help="Type of file to plot from. 0 - from lheAnalyzer, 1 - from cofeGeno")

opt = parser.parse_args()

lumi = 1 #fb
scales_150_250 = {
    "2016_nj":
    {
        "xs": 100, #230 <-- there is a problem with normalization
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

nanogen_scales_250_400 = {
    "2016_nj":
    {
        "xs": 5.0,#2.848,
        "fneg": 0.33,
        "Nev": 85600,
    },
    "2017_1j": 
    {
        "xs": 1.27,
        "fneg": 0.14,
        "Nev": 30000,
    },
    "2017_2j":
    {
        "xs": 2.958,
        "fneg": 0.34,
        "Nev": 67100,
     },
}


def plotAll(inputType, scales, hAll, h2016_nj=None, h2017_1j=None, h2017_2j=None):

    if inputType==0:
        observables = h2016_nj.keys()
    elif inputType==1:
        observables = hAll.keys()

    for obs in observables:
        print(obs)
        if obs in ['wei','lep_eta','z_mass','jet_pt','dijet_pt','nlep','njet','njet15','cutflow','sumw']: continue 
        if inputType==0:
            h1 = h2016_nj[obs]
            h2 = h2017_1j[obs]
            h3 = h2017_2j[obs]
            plotType0(obs, h1,h2,h3, scales)
        elif inputType==1:
            plotType1(obs, hAll[obs], scales)
        
def plotType0(obs, h2016_nj,h2017_1j,h2017_2j, scales):
    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(outdir+'_scaled'):
        os.makedirs(outdir+'_scaled')


    h_2016_nj = h2016_nj*scales["2016_nj"]["xs"]*1000*lumi/(1-2*scales["2016_nj"]["fneg"])/scales["2016_nj"]["Nev"]
    h_2017_1j = h2017_1j*scales["2017_1j"]["xs"]*1000*lumi/(1-2*scales["2017_1j"]["fneg"])/scales["2017_1j"]["Nev"]
    h_2017_2j = h2017_2j*scales["2017_2j"]["xs"]*1000*lumi/(1-2*scales["2017_2j"]["fneg"])/scales["2017_2j"]["Nev"]
    plt.gcf().clf()
    h_2016_nj.plot(label='2016 1+2j')
    h_2017_1j.plot(label='2017 1j')
    h_2017_2j.plot(label='2017 2j')
    plt.legend(prop={'size': 10})
    plt.gcf().savefig(f"{outdir}/{obs}.png")
    
    # print(type(h_2017_1j))
    h_2017_nj =  h_2017_1j +  h_2017_2j
    plt.gcf().clf()
    
    h_2016_nj.plot_ratio(h_2017_nj,     
                         rp_num_label="2016 1+2j",
                         rp_denom_label="2017 1+2j",     
                         rp_uncert_draw_type="line",
                         rp_uncertainty_type="poisson",
                         rp_ylim=[0.2, 2.2],
                     )
    
    plt.gcf().savefig(f"{outdir}_scaled/{obs}.png")


def plotType1(obs, hAll, scales):

    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(outdir+'_scaled'):
        os.makedirs(outdir+'_scaled')

    sc = {
        '2016_DYnJ': scales["2016_nj"]["xs"]*1000*lumi/(1-2*scales["2016_nj"]["fneg"])/scales["2016_nj"]["Nev"],
        '2017_DY1J': scales["2017_1j"]["xs"]*1000*lumi/(1-2*scales["2017_1j"]["fneg"])/scales["2017_1j"]["Nev"],
        '2017_DY2J': scales["2017_2j"]["xs"]*1000*lumi/(1-2*scales["2017_2j"]["fneg"])/scales["2017_2j"]["Nev"],
    }
    print(hAll)


    plt.gcf().clf()
    hist.plot1d(hAll, overlay="dataset", line_opts={})
    plt.gcf().savefig(f"{outdir}/{obs}.png")

    hAll.scale(sc, axis='dataset')

    hNew = hAll.group("dataset", hist.Cat("sample", "Merged whole year"), 
                      { "2016 1+2j": ["2016_DYnJ"],
                        "2017 1+2j": ["2017_DY1J", "2017_DY2J"]})
    print(hNew)

    fig, (ax, rax) = plt.subplots(nrows=2, ncols=1, figsize=(7,7), 
                                  gridspec_kw={"height_ratios": (3, 1)},sharex=True)
    fig.subplots_adjust(hspace=.07)
    

    hist.plot1d(hNew, overlay="sample", ax=ax, line_opts={})
    ax.set_ylim(0, None)

    leg = ax.legend()

    hist.plotratio(num = hNew["2016 1+2j"].project(obs),
                   denom = hNew["2017 1+2j"].project(obs),
                   error_opts={'color': 'k', 'marker': '.'},
                   ax=rax,
                   denom_fill_opts={},
                   guide_opts={},
                   unc='num'
               )
    
    rax.set_ylabel('Ratio')
    rax.set_ylim(0,2)
    plt.gcf().savefig(f"{outdir}_scaled/{obs}.png")
    
    
def readFromPickles(inputfile):
    hists = pkl.load(open(inputfile,'rb'))

    return hists


if __name__ == "__main__":
    print("This is the __main__ part")
    
    if opt.inputType==0:

        scales = scales_150_250
        h2016_nj = readFromPickles('plots_2016_nj_ZpT_160_240/Pickles.pkl')
        h2017_1j = readFromPickles('plots_2017_1j_ZpT_160_240/Pickles.pkl')
        h2017_2j = readFromPickles('plots_2017_2j_ZpT_160_240/Pickles.pkl')
        #scales = scales_250_400
        #h2016_nj = readFromPickles('plots_2016_nj_ZpT_260_390/Pickles.pkl')
        #h2017_1j = readFromPickles('plots_2017_1j_ZpT_260_390/Pickles.pkl')
        #h2017_2j = readFromPickles('plots_2017_2j_ZpT_260_390/Pickles.pkl')
        
        plotAll(opt.inputType, scales, None, h2016_nj, h2017_1j, h2017_2j)
        
    elif opt.inputType==1:
        scales = nanogen_scales_250_400
        #pickledCoffea = readFromPickles('plots_CoffeaHist/Pickles.pkl')
        pickledCoffea = readFromPickles('plots_Cofiano_260_390/Pickles.pkl')
        plotAll(opt.inputType, scales, pickledCoffea)
