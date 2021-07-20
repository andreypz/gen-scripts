#!/usr/bin/env python3

import os

import numpy as np
from hist import Hist
from lhereader import LHEReader
from matplotlib import pyplot as plt


import argparse
parser = argparse.ArgumentParser(description='Run quick plots', usage="./xx lhefile")
parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
parser.add_argument("lhefile")

opt = parser.parse_args()


def plot(histograms):
    '''Plots all histograms. No need to change.'''
    outdir = opt.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for observable, histogram in histograms.items():
        plt.gcf().clf()
        histogram.plot()
        plt.gcf().savefig(f"{outdir}/{observable}.png")

def setup_histograms():
    '''Histogram initialization. Add new histos here.'''

    # Bin edges for each observable
    bins ={
        'wei'        : np.linspace(-50,50,100),
        'nlep'       : np.linspace(0,3,6),
        'lep_eta'    : np.linspace(-5,5,50),
        'lep_pt'     : np.linspace(0,500,50),
        'dilep_m'    : np.linspace(50,120,50),
        'dilep_pt'   : np.linspace(0,600,100),

        'z_mass'     : np.linspace(50,120,50),
        'z_pt'       : np.linspace(0,600,100),
        'njet'       : np.linspace(0,3,6),
        'njet10'     : np.linspace(0,3,6),
        'jet_eta'    : np.linspace(-5,5,50),
        'jet_pt'     : np.linspace(0,500,50),
        'j_ht'       : np.linspace(0,500,50),
        'dijet_m'    : np.linspace(0,600,50),
        'dijet_pt'   : np.linspace(0,600,100),
        'dijet_dr'   : np.linspace(0,5,50),

    } 

    histograms = { 
                    observable : (
                                    Hist.new
                                    .Var(binning, name=observable, label=observable)
                                    .Double()
                                )
                    for observable, binning in bins.items()
    }

    return histograms

def analyze(lhe_file):
    '''Event loop + histogram filling'''

    reader = LHEReader(
        lhe_file,
        #weight_mode='dict',         # Weights will be read as a dictionary
        #weight_regex='(1|.*Coup.*)' # Only read weights with ID 0 (nominal),
        # or fitting our reweight names
    )
    histograms = setup_histograms()
    for event in reader:
        # Find charged leptons
        #leptons = filter(
        #                lambda x: abs(x.pdgid) in (11,13,15),
        #                event.particles
        #                )

        #Zs = filter(
        #        lambda x: x.pdgid==23,
        #        event.particles
        #    )


        leptons = [x for x in event.particles if abs(x.pdgid) in (11,13,15)]
        Zs = [x for x in event.particles if x.pdgid==23]
        jets = [x for x in event.particles if abs(x.pdgid) in (1,2,3,4,5,21) and x.status==1]
        jets10 = [x for x in event.particles if abs(x.pdgid) in (1,2,3,4,5,21) and x.status==1 and x.p4().pt>10]

        nlep = len(leptons)
        njet = len(jets)
        njet10 = len(jets10)
        histograms['nlep'].fill(nlep, weight=1)
        histograms['njet'].fill(njet, weight=1)
        histograms['njet10'].fill(njet10, weight=1)

        if nlep!=2:
            print("nlep=%i"%nlep)
            continue

        # Sum over all lepton four-momenta in the event
        v_p4 = None
        for l in leptons:
            if l.p4().pt<1: continue
            if v_p4:
                v_p4 += l.p4()
            else:
                v_p4 = l.p4()

        
        vpt = v_p4.pt
        vmass = v_p4.mass
        
        if vpt<250 or vpt>400: continue
        if vmass<60 or vmass>120: continue

        wei = event.weights[0]
        #wei = 1
        #print(wei)
        if abs(wei) not in (11.397, 11.396):
            print(wei)
            continue
        histograms['wei'].fill(wei, weight=1)


        histograms['dilep_m'].fill(vmass, weight=wei)
        histograms['dilep_pt'].fill(vpt, weight=wei)

        for l in leptons:
            #print("Lep pt = ", l.p4().pt, "eta=", l.p4().eta)
            histograms['lep_eta'].fill(l.p4().eta, weight=wei)
            histograms['lep_pt'].fill(l.p4().pt, weight=wei)


        for Z in Zs:
            #print("Z pt = ", Z.p4().pt, "eta=", Z.p4().eta)
            zpt = Z.p4().pt
            zmass = Z.p4().mass
            histograms['z_mass'].fill(zmass, weight=wei)
            histograms['z_pt'].fill(zpt, weight=wei)


        j_p4 = None
        for j in jets:
            histograms['jet_eta'].fill(j.p4().eta, weight=wei)
            histograms['jet_pt'].fill(j.p4().pt, weight=wei)
            if j_p4:
                j_p4 += j.p4()
            else:
                j_p4 = j.p4()


        histograms['j_ht'].fill(j_p4.pt, weight=wei)

        if njet10>=2:
            j1 = jets10[0].p4()
            j2 = jets10[1].p4()
            dijet_pt = (j1+j2).pt
            dijet_mass = (j1+j2).mass
            if dijet_pt<250 or dijet_pt>400: continue
            histograms['dijet_dr'].fill(j1.deltar(j2), weight=wei)
            histograms['dijet_m'].fill(dijet_mass, weight=wei)
            histograms['dijet_pt'].fill(dijet_pt, weight=wei)

    return histograms





if __name__ == "__main__":
    print("This is the __main__ part")

    histograms = analyze(opt.lhefile)
    plot(histograms)