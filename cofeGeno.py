#!/usr/bin/env python3

from os import listdir, makedirs, path, system
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from coffea import hist
import coffea.processor as processor
import awkward as ak
from coffea.nanoevents import NanoEventsFactory

from Coffea_NanoGEN_schema import NanoGENSchema


def getRootFiles(d, lim=None):
    if "xrootd" in d:
        import subprocess
        sp = d.split("/")
        siteIP = "/".join(sp[0:3])
        pathToFiles = "/".join(sp[3:-1])
        allfiles = str(subprocess.check_output(["xrdfs", siteIP, "ls", pathToFiles]), 'utf-8').split("\n")
        rootfiles = [siteIP+f for i,f in enumerate(allfiles) if f.endswith(".root") and (lim==None or i<lim)]
    else:
        rootfiles = [path.join(d, f) for i,f in enumerate(listdir(d)) if f.endswith(".root") and (lim==None or i<lim)]
    # print(rootfiles)
    return rootfiles
class Processor(processor.ProcessorABC):
    def __init__(self):
        
        axis = { "dataset": hist.Cat("dataset", ""),
                 "LHE_Vpt": hist.Bin("LHE_Vpt", "V PT [GeV]", 100, 0, 600),                 
                 'wei'        : hist.Bin("wei", "wei", 50, -10, 10), 
                 'nlep'       : hist.Bin("nlep", "nlep", 12, 0, 6), 
                 'lep_eta'    : hist.Bin("lep_eta", "lep_eta", 50, -5, 5), 
                 'lep_pt'     : hist.Bin("lep_pt", "lep_pt", 50, 0, 500), 
                 'dilep_m'    : hist.Bin("dilep_m", "dilep_m", 50, 50, 120), 
                 'dilep_pt'   : hist.Bin("dilep_pt", "dilep_pt", 100, 0, 600), 
                 'njet15'     : hist.Bin("njet15", "njet15", 12, 0, 6), 
                 'jet_eta'    : hist.Bin("jet_eta", "jet_eta", 50, -5, 5), 
                 'jet_pt'     : hist.Bin("jet_pt", "jet_pt", 50, 0, 500), 
                 'dijet_dr'   : hist.Bin("dijet_dr", "dijet_dr", 50, 0, 5), 
                 'dijet_m'    : hist.Bin("dijet_m", "dijet_m", 50, 0, 1200), 
                 'dijet_pt'   : hist.Bin("dijet_pt", "dijet_pt", 100, 0, 600)
             }
        
        self._accumulator = processor.dict_accumulator( 
            {observable : hist.Hist("Counts", axis["dataset"], var_axis) for observable, var_axis in axis.items() if observable!="dataset"}
        )
        self._accumulator['cutflow'] = processor.defaultdict_accumulator(int)
        
    
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        output = self.accumulator.identity()
        #print(output)

        dataset = events.metadata["dataset"]
        LHE_Vpt = events.LHE['Vpt']
        #print(LHE_Vpt)
        # We can define a new key for cutflow (in this case 'all events'). 
        # Then we can put values into it. We need += because it's per-chunk (demonstrated below)
        output['cutflow']['all events'] += ak.size(LHE_Vpt)
        output['cutflow']['number of chunks'] += 1
        

        particles = events.LHEPart

        leptons = particles[ (np.abs(particles.pdgId) == 11) | (np.abs(particles.pdgId) == 13) | (np.abs(particles.pdgId) == 15) ]
        jets15  = particles[ ( (np.abs(particles.pdgId) == 1) | (np.abs(particles.pdgId) == 2) | (np.abs(particles.pdgId) == 3 ) |
                               (np.abs(particles.pdgId) == 4) | (np.abs(particles.pdgId) == 5) | (np.abs(particles.pdgId) == 21 ) ) &
                             (particles.status==1) & (particles.pt > 15) ]

        weight_nosel = events.genWeight
        output['LHE_Vpt'].fill(dataset=dataset, LHE_Vpt=LHE_Vpt, weight=weight_nosel)

        output['wei'].fill(dataset=dataset, wei=weight_nosel/np.abs(weight_nosel))
        
        LL_events = events[ak.num(leptons) == 2]
        JJ_events = events[ak.num(jets15) >= 2]

        output['nlep'].fill(dataset=dataset, nlep=ak.num(leptons))
        output['njet15'].fill(dataset=dataset, njet15=ak.num(jets15))

        two_lep = ak.num(leptons) == 2
        zLL = leptons[two_lep][:, 0] + leptons[two_lep][:, 1]
        vpt = zLL.pt
        vmass = zLL.mass
        
        vpt_cut =  (vpt>=260) & (vpt<=390)
        vmass_cut = (vmass>=60) & (vmass<=120)

        two_jets = ak.num(jets15) >= 2
       
        full_selection = two_lep & two_jets & vpt_cut & vmass_cut

        selected_events = events[full_selection]
        output['cutflow']['seleced_events'] += len(selected_events)

        j_2l2j = jets15[full_selection]
        dijet = j_2l2j[:, 0] + j_2l2j[:, 1]    

        dijet_pt = dijet.pt
        dijet_m  = dijet.mass
        dijet_dr = j_2l2j[:, 0].delta_r(j_2l2j[:, 1])
        

        weight = selected_events.genWeight
        #weight = np.ones(len(selected_events))

        output['dilep_m'].fill(dataset=dataset, dilep_m=vmass[full_selection], weight=weight)
        output['dilep_pt'].fill(dataset=dataset, dilep_pt=vpt[full_selection], weight=weight)
        
        output['lep_eta'].fill(dataset=dataset, lep_eta=ak.flatten(leptons.eta[full_selection]))
        output['lep_pt'].fill(dataset=dataset, lep_pt=ak.flatten(leptons.pt[full_selection]))
        
        output['jet_eta'].fill(dataset=dataset, jet_eta=ak.flatten(jets15.eta[full_selection]))
        output['jet_pt'].fill(dataset=dataset, jet_pt=ak.flatten(jets15.pt[full_selection]))
        
        output['dijet_dr'].fill(dataset=dataset, dijet_dr=dijet_dr, weight=weight)
        output['dijet_m'].fill(dataset=dataset, dijet_m=dijet_m, weight=weight)
        output['dijet_pt'].fill(dataset=dataset, dijet_pt=dijet_pt, weight=weight)

        return output

    def postprocess(self, accumulator):
        return accumulator



def plot(histograms, outdir, fromPickles=False):
    '''Plots all histograms. No need to change.'''
    if not path.exists(outdir):
        makedirs(outdir)

    for observable, histogram in histograms.items():
        #print (observable, histogram, type(histogram))
        if type(histogram) is hist.hist_tools.Hist:
            print(observable, "I am a Hist", histogram)
        else:
            continue
        plt.gcf().clf()
        #hist.plot1d(histogram, overlay='dataset', fill_opts={'edgecolor': (0,0,0,0.3), 'alpha': 0.8}, line_opts={})
        hist.plot1d(histogram, overlay='dataset', line_opts={}, overflow='none')
        #hproj = histogram["2016_DYnJ"]
        #hproj = histogram["2017_DY1J"]
        #hproj = histogram["2017_DY2J"]
        #hist.plot1d(hproj, line_opts={}, overflow='none')
        plt.gca().autoscale()
        plt.gcf().savefig(f"{outdir}/{observable}.png")

    if not fromPickles:
        pkl.dump( histograms,  open(outdir+'/Pickles.pkl',  'wb')  )

def plotFromPickles(inputfile, outdir):
    hists = pkl.load(open(inputfile,'rb'))
    plot(hists, outdir, True)

if __name__ == "__main__":
    print("This is the __main__ part")

    import argparse
    parser = argparse.ArgumentParser(description='Run quick plots from NanoGEN input files')
    #parser.add_argument("inputfile")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument('--pkl', type=str, default=None,  help="Make plots from pickled file.")

    opt = parser.parse_args()

    print(opt)

    #from dask.distributed import Client
    import time
    
    #client = Client("tls://localhost:8786")
    ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/NanoGEN/"
    #ntuples_location = "/net/data_cms/institut_3a/NanoGEN/"
    p2016_DYn_LHE_250_400 = ntuples_location + "/DYnJetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Summer15/FromGridPack-12Aug2021/210812_100639/0000/"
    p2017_DY1_LHE_250_400 = ntuples_location + "/DY1JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-12Aug2021/210812_100210/0000/"
    p2017_DY2_LHE_250_400 = ntuples_location + "/DY2JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-12Aug2021/210812_100403/0000/"
    file_list = {
        '2016_DYnJ' :  getRootFiles(p2016_DYn_LHE_250_400),
        '2017_DY1J' :  getRootFiles(p2017_DY1_LHE_250_400),
        '2017_DY2J' :  getRootFiles(p2017_DY2_LHE_250_400),
        #'2017_DY1J' :  [p2017_DY1_LHE_250_400+"/Tree_1.root"],
        #'2017_DY2J' :  [p2017_DY2_LHE_250_400+"/Tree_1.root"],
        #'2016_DYnJ' :  [p2016_DYn_LHE_250_400+"/Tree_1.root"],
    }
    
    if opt.pkl!=None:
        plotFromPickles(opt.pkl, opt.outdir)
    else:
        output = processor.run_uproot_job(file_list,
                                          treename = 'Events',
                                          processor_instance = Processor(),
                                          #executor = processor.iterative_executor,
                                          executor = processor.futures_executor,
                                          executor_args = {'schema': NanoGENSchema, "workers":8}
                                      )
        
        
        
        plot(output, opt.outdir)
    
    
        for key, value in output['cutflow'].items():
            print(key, value)

        
        
