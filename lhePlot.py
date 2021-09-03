#!/usr/bin/env python3

import os

import numpy as np
from hist import Hist
from lhereader import LHEReader
from matplotlib import pyplot as plt
import pickle as pkl

def plot(histograms, outdir, fromPickles=False):
    '''Plots all histograms. No need to change.'''
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for observable, histogram in histograms.items():
        plt.gcf().clf()
        histogram.plot()
        plt.gcf().savefig(f"{outdir}/{observable}.png")

    if not fromPickles:
        pkl.dump( histograms,  open(outdir+'/Pickles.pkl',  'wb')  )

def plotFromPickles(inputfile, outdir):
    hists = pkl.load(open(inputfile,'rb'))
    # print(hists)
    plot(hists, outdir, True)

def setup_histograms():
    '''Histogram initialization. Add new histos here.'''

    # Bin edges for each observable
    bins ={
        'wei'        : np.linspace(-5,5,51),
        'nlep'       : np.linspace(0,6,13),
        'lep_eta'    : np.linspace(-5,5,51),
        'lep_pt'     : np.linspace(0,500,51),
        'dilep_m'    : np.linspace(50,120,51),
        'dilep_pt'   : np.linspace(0,600,101),

        'z_mass'     : np.linspace(50,120,51),
        'z_pt'       : np.linspace(0,600,101),
        'njet'       : np.linspace(0,6,13),
        'njet15'     : np.linspace(0,6,13),
        'jet_eta'    : np.linspace(-5,5,51),
        'jet_pt'     : np.linspace(0,500,51),
        'jets_ht'    : np.linspace(0,500,51),
        'dijet_m'    : np.linspace(0,1200,51),
        'dijet_pt'   : np.linspace(0,600,101),
        'dijet_dr'   : np.linspace(0,5,51),

    } 

    histograms = { 
                    observable : (
                                    Hist.new
                                    .Var(binning, name=observable, label=observable)
                                    .Weight()
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
        jets15 = [x for x in event.particles if abs(x.pdgid) in (1,2,3,4,5,21) and x.status==1 and x.p4().pt>15]

        nlep = len(leptons)
        njet = len(jets)
        njet15 = len(jets15)
        histograms['nlep'].fill(nlep, weight=1)
        histograms['njet'].fill(njet, weight=1)
        histograms['njet15'].fill(njet15, weight=1)

        if nlep!=2:
            print("Not 2L channel!  nlep=%i"%nlep)
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

        wei = event.weights[0]
        #wei = 1
        #print(wei)
        histograms['wei'].fill(wei/abs(wei), weight=1)
        
        
        #if vpt<160 or vpt>240: continue
        if vpt<260 or vpt>390: continue
        if vmass<60 or vmass>120: continue
        if njet15<2: continue


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


        histograms['jets_ht'].fill(j_p4.pt, weight=wei)

        if njet15>=2:
            j1 = jets15[0].p4()
            j2 = jets15[1].p4()
            dijet_dr = j1.deltar(j2)
            dijet_pt = (j1+j2).pt
            dijet_mass = (j1+j2).mass

            histograms['dijet_dr'].fill(dijet_dr, weight=wei)
            histograms['dijet_m'].fill(dijet_mass, weight=wei)
            histograms['dijet_pt'].fill(dijet_pt, weight=wei)

    return histograms





if __name__ == "__main__":
    print("This is the __main__ part")

    import argparse
    parser = argparse.ArgumentParser(description='Run quick plots from LHE input files')
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument("inputfile")
    
    opt = parser.parse_args()



    if opt.inputfile.endswith(".lhe"):
        histograms = analyze(opt.inputfile)
        plot(histograms, opt.outdir)
    elif  opt.inputfile.endswith(".pkl"):
        plotFromPickles(opt.inputfile, opt.outdir)
