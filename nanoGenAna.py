#!/usr/bin/env python3

import numpy as np
from hist import Hist
from matplotlib import pyplot as plt
import pickle as pkl
import uproot as up
from lhePlot import setup_histograms, plot
from skhep.math import LorentzVector
import awkward as ak
from coffea.nanoevents import NanoEventsFactory
from Coffea_NanoGEN_schema import NanoGENSchema

def loop(in_file):
    histograms = setup_histograms()
    
    events = NanoEventsFactory.from_root(in_file, schemaclass=NanoGENSchema).events()
    
    print(events.fields)
    print(events.GenPart.fields)
    print(events.LHEPart.fields)

    LL_events = 0
    JJ_events = 0
    selected_events = 0
    for i,ev in enumerate(events):
        if i < 10: 
            print(i, ev.event, ev.run)
        #leptons = [x for x in ev.GenPart if abs(x.pdgId) in (11,13,15)]
        leptons = [x for x in ev.LHEPart if abs(x.pdgId) in (11,13,15)]
        #Zs = [x for x in ev.LHEPart if x.pdgId==23]
        jets15 = [x for x in ev.LHEPart if abs(x.pdgId) in (1,2,3,4,5,21) and x.status==1 and x.pt>15]

        nlep = len(leptons)
        njet15 = len(jets15)
        histograms['nlep'].fill(nlep, weight=1)
        histograms['njet15'].fill(njet15, weight=1)

        if nlep==2:
            LL_events += 1
        if njet15>=2:
            JJ_events+=1


        v_p4 = None
        for l in leptons:
            #print(l)
            l_p4 = LorentzVector(l.x, l.y, l.z, l.t)
            if l.pt<1: continue
            if v_p4:
                v_p4 += l_p4
            else:
                v_p4 = l_p4


        wei = ev.genWeight
        #print(wei)
        histograms['wei'].fill(wei/abs(wei), weight=1)
        #wei = 1


        if nlep!=2:
            print("Not 2L channel!  nlep=%i"%nlep)
            continue

        vpt = v_p4.pt
        vmass = v_p4.mass

        if vpt<260 or vpt>390: continue
        if vmass<60 or vmass>120: continue
        if njet15<2: continue

        histograms['dilep_m'].fill(vmass, weight=wei)
        histograms['dilep_pt'].fill(vpt, weight=wei)

        for l in leptons:
            #print("Lep pt = ", l.p4().pt, "eta=", l.p4().eta)
            histograms['lep_eta'].fill(l.eta, weight=1)
            histograms['lep_pt'].fill(l.pt, weight=1)


        #for Z in Zs:
        #    print("Z pt = ", Z.pt, "eta=", Z.eta)
        #    zpt = Z.pt
        #    zmass = Z.mass
        #    histograms['z_mass'].fill(zmass, weight=wei)
        #    histograms['z_pt'].fill(zpt, weight=wei)

        j_p4 = None
        for j in jets15:
            histograms['jet_eta'].fill(j.eta, weight=1)
            histograms['jet_pt'].fill(j.pt, weight=1)
            if j_p4:
                j_p4 += LorentzVector(j.x, j.y, j.z, j.t)
            else:
                j_p4 = LorentzVector(j.x, j.y, j.z, j.t)

        histograms['jets_ht'].fill(j_p4.pt, weight=wei)

        if njet15>=2:
            j1 = jets15[0]
            j2 = jets15[1]
            dijet_dr = j1.delta_r(j2)
            dijet_pt = (j1+j2).pt
            dijet_mass = (j1+j2).mass

            histograms['dijet_dr'].fill(dijet_dr, weight=wei)
            histograms['dijet_m'].fill(dijet_mass, weight=wei)
            histograms['dijet_pt'].fill(dijet_pt, weight=wei)
            
            selected_events += 1


    print("Events with two leptons:", LL_events )
    print("Events with two jets:", JJ_events )
    print("Finals events:", selected_events)
    print("Loop finished")

    return histograms

def coffea(in_file):
    print("Beware: coffeine is a drug!")
    histograms = setup_histograms()

    events = NanoEventsFactory.from_root(in_file, schemaclass=NanoGENSchema).events()

    particles = events.LHEPart

    leptons = particles[ (np.abs(particles.pdgId) == 11) | (np.abs(particles.pdgId) == 13) | (np.abs(particles.pdgId) == 15) ]
    jets15  = particles[ ( (np.abs(particles.pdgId) == 1) | (np.abs(particles.pdgId) == 2) | (np.abs(particles.pdgId) == 3 ) |
                           (np.abs(particles.pdgId) == 4) | (np.abs(particles.pdgId) == 5) | (np.abs(particles.pdgId) == 21 ) ) &
                         (particles.status==1) & (particles.pt > 15) ]
    
    LL_events = events[ak.num(leptons) == 2]
    JJ_events = events[ak.num(jets15) >= 2]
    histograms['nlep'].fill(ak.num(leptons), weight=1)
    histograms['njet15'].fill(ak.num(jets15), weight=1)

    zLL = leptons[:, 0] + leptons[:, 1]
    dijet = jets15[:, 0] + jets15[:, 1]

    vpt = zLL.pt
    vmass = zLL.mass

    dijet_pt = dijet.pt
    dijet_mass = dijet.mass
    dijet_dr = jets15[:, 0].delta_r(jets15[:, 1])


    two_lep = ak.num(leptons) == 2
    two_jets = ak.num(jets15) >= 2
    vpt_cut =  (vpt>=260) & (vpt<=390) 
    vmass_cut = (vmass>=60) & (vmass<=120)

    full_selection = two_lep & two_jets & vpt_cut & vmass_cut

    selected_events = events[two_lep & two_jets & vpt_cut & vmass_cut]
    
    wei_nosel = events.genWeight
    wei = events[full_selection].genWeight
    #wei = 1
    #print(wei, len(wei))
    
    histograms['wei'].fill(wei_nosel/np.abs(wei_nosel), weight=1)

    histograms['dilep_m'].fill(vmass[full_selection], weight=wei)
    histograms['dilep_pt'].fill(vpt[full_selection], weight=wei)


    histograms['lep_eta'].fill(ak.flatten(leptons.eta[full_selection]), weight=1)
    histograms['lep_pt'].fill(ak.flatten(leptons.pt[full_selection]), weight=1)

    histograms['jet_eta'].fill(ak.flatten(jets15.eta[full_selection]), weight=1)
    histograms['jet_pt'].fill(ak.flatten(jets15.pt[full_selection]), weight=1)    


    histograms['dijet_dr'].fill(dijet_dr[full_selection], weight=wei)
    histograms['dijet_m'].fill(dijet_mass[full_selection], weight=wei)
    histograms['dijet_pt'].fill(dijet_pt[full_selection], weight=wei)

    histograms['jets_ht'].fill(dijet_pt[full_selection], weight=wei)    


    histograms['LHE_Vpt'] = Hist.new.Var(np.linspace(0,600,100), name="LHE_Vpt", label="LHE_Vpt").Weight()
    histograms['LHE_Vpt'].fill(events[full_selection].LHE["Vpt"], weight=wei)

    print(events.LHE, events[full_selection].LHE["Vpt"])

    print("Events with two leptons:", len(LL_events) )
    print("Events with two jets:", len(JJ_events) )
    print("Finals events:", len(selected_events) )
    print("Coffea finished")
    
    return histograms


if __name__ == "__main__":
    print("This is the __main__ part")

    import argparse
    parser = argparse.ArgumentParser(description='Run quick plots from NanoGEN input files')
    parser.add_argument("inputfile")
    parser.add_argument('-l','--loop', default=False,  action="store_true", help="Run the loop version of the code")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    
    opt = parser.parse_args()
    
    print(opt)



    if opt.loop:
        hists = loop(opt.inputfile)
    else:
        hists = coffea(opt.inputfile)

    plot(hists, opt.outdir)
