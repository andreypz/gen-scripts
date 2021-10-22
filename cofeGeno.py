#!/usr/bin/env python3

from os import listdir, makedirs, path, system
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from coffea import hist
import coffea.processor as processor
import awkward as ak
from coffea.nanoevents import NanoEventsFactory
from functools import partial

from Coffea_NanoGEN_schema import NanoGENSchema
import sampleInfo as si

def isClean(obj_A, obj_B, drmin=0.4):
    # From: https://github.com/oshadura/topcoffea/blob/master/topcoffea/modules/objects.py
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

class Processor(processor.ProcessorABC):
    def __init__(self):
        
        axis = { "dataset": hist.Cat("dataset", ""),
                 "LHE_Vpt": hist.Bin("LHE_Vpt", "LHE V PT [GeV]", 100, 0, 600),                 
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
        self._accumulator['cutflow'] = processor.defaultdict_accumulator( partial(processor.defaultdict_accumulator, int) )
        self._accumulator["sumw"] =  processor.defaultdict_accumulator( float ) 
     
    
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
        output['cutflow'][dataset]['all_events'] += ak.size(LHE_Vpt)
        output['cutflow'][dataset]['number_of_chunks'] += 1
        
        particles = events.GenPart
        #particles = events.LHEPart
        leptons = particles[ ((np.abs(particles.pdgId) == 11) | (np.abs(particles.pdgId) == 13) ) &
                             (particles.status == 1) & (particles.pt>15) & (np.abs(particles.eta)<2.5) ]
        
        genjets = events.GenJet
        jets15 = genjets[ (np.abs(genjets.eta) < 2.5)  &  (genjets.pt > 25) ]
        #jets15 = particles[ (np.abs(particles.pdgId) == 11) | (np.abs(particles.pdgId) == 13) | (np.abs(particles.pdgId) == 15) ]
        #jets15  = particles[ ( (np.abs(particles.pdgId) == 1) | (np.abs(particles.pdgId) == 2) | (np.abs(particles.pdgId) == 3 ) |
        #                       (np.abs(particles.pdgId) == 4) | (np.abs(particles.pdgId) == 5) | (np.abs(particles.pdgId) == 21 ) ) &
        #                     (particles.status==1) & (particles.pt > 15) ]


        
        weight_nosel = events.genWeight
        output['LHE_Vpt'].fill(dataset=dataset, LHE_Vpt=LHE_Vpt, weight=weight_nosel)
        
        output["sumw"][dataset] += np.sum(weight_nosel)
        print(dataset, weight_nosel)
        
        output['wei'].fill(dataset=dataset, wei=weight_nosel/np.abs(weight_nosel))
        
        #LL_events = events[ak.num(leptons) == 2]
        #JJ_events = events[ak.num(jets15) >= 2]
        
        output['nlep'].fill(dataset=dataset, nlep=ak.num(leptons))
        output['njet15'].fill(dataset=dataset, njet15=ak.num(jets15))


        dileptons = ak.combinations(leptons, 2, fields=['i0', 'i1'])

        pt25  = ((dileptons['i0'].pt > 25) | (dileptons['i1'].pt > 25))
        Zmass_cut = (((dileptons['i0'] + dileptons['i1']).mass - 91.19) < 15)
        Vpt_cut = ( (dileptons['i0'] + dileptons['i1']).pt > 100)
        dileptonMask = pt25 & Zmass_cut
        good_dileptons = dileptons[dileptonMask]
        
        vpt = (good_dileptons['i0'] + good_dileptons['i1']).pt
        vmass = (good_dileptons['i0'] + good_dileptons['i1']).mass

        two_lep = ak.num(good_dileptons) == 1
        

        LHE_vpt_cut = (LHE_Vpt>=155) & (LHE_Vpt<=245)
        #LHE_vpt_cut = (LHE_Vpt>=260) & (LHE_Vpt<=390)


        jets15['isClean'] = isClean(jets15, leptons, drmin=0.5)
        j_isclean = isClean(jets15, leptons, drmin=0.5)

        #good_jets = jets
        good_jets = jets15[j_isclean]
        two_jets = (ak.num(good_jets) >= 2)
        
        full_selection = two_lep & two_jets & LHE_vpt_cut
        #full_selection = two_lep & two_jets & LHE_vpt_cut & vmass_cut
        #full_selection = two_lep & two_jets & vpt_cut & vmass_cut
        
        selected_events = events[full_selection]
        output['cutflow'][dataset]["selected_events"] += len(selected_events)


        dijets = good_jets[full_selection]
        dijet = dijets[:, 0] + dijets[:, 1]

        dijet_pt = dijet.pt
        dijet_m  = dijet.mass
        dijet_dr = dijets[:, 0].delta_r(dijets[:, 1])
        
        
        weight = selected_events.genWeight
        #weight = np.ones(len(selected_events))
        
        output['dilep_m'].fill(dataset=dataset, dilep_m=ak.flatten(vmass[full_selection]), weight=weight)
        output['dilep_pt'].fill(dataset=dataset, dilep_pt=ak.flatten(vpt[full_selection]), weight=weight)
        
        output['lep_eta'].fill(dataset=dataset, lep_eta=ak.flatten(leptons.eta[full_selection]))
        output['lep_pt'].fill(dataset=dataset, lep_pt=ak.flatten(leptons.pt[full_selection]))
        
        output['jet_eta'].fill(dataset=dataset, jet_eta=ak.flatten(good_jets.eta[full_selection]))
        output['jet_pt'].fill(dataset=dataset, jet_pt=ak.flatten(good_jets.pt[full_selection]))
        
        output['dijet_dr'].fill(dataset=dataset, dijet_dr=dijet_dr, weight=weight)
        output['dijet_m'].fill(dataset=dataset, dijet_m=dijet_m, weight=weight)
        output['dijet_pt'].fill(dataset=dataset, dijet_pt=dijet_pt, weight=weight)

        return output

    def postprocess(self, accumulator):

        lumi = 11 # random lumi, it does not matter here
        
        print(accumulator['sumw'])

        weights = { '2016_DYnJ': lumi*8.47/accumulator['sumw']['2016_DYnJ'],
                    '2017_DY1J': lumi*8.47/accumulator['sumw']['2017_DY1J'],
                    '2017_DY2J': lumi*8.47/accumulator['sumw']['2017_DY2J'],
                }
        print(weights)

        scaled = hist.Cat('ds_scaled', 'ds_scaled')
        for key in accumulator:
            if key not in ['cutflow','sumw']:
                accumulator[key].scale(weights, axis='dataset')
                
                accumulator[key] = accumulator[key].group('dataset', scaled, {'2016_DY 1+2j': ['2016_DYnJ'], 
                                                                              '2017_DY 1+2j': ['2017_DY1J', '2017_DY2J'],
                                                                          })
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

        #print(histogram.axes())
        #print(list(map(lambda x:x.name, histogram.axes() )))
        axes = list(map(lambda x:x.name, histogram.axes() ))
        if 'dataset' in axes:
            hist.plot1d(histogram, overlay='dataset', line_opts={}, overflow='none')
        elif 'ds_scaled' in axes:
            hist.plot1d(histogram, overlay='ds_scaled', line_opts={}, overflow='none')
        plt.gca().autoscale()
        plt.gcf().savefig(f"{outdir}/{observable}.png")

    if not fromPickles:
        pkl.dump( histograms,  open(outdir+'/Pickles.pkl',  'wb')  )

def plotFromPickles(inputfile, outdir):
    hists = pkl.load(open(inputfile,'rb'))
    plot(hists, outdir, True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run quick plots from NanoGEN input files')
    #parser.add_argument("inputfile")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument('--pkl', type=str, default=None,  help="Make plots from pickled file.")
    parser.add_argument('-n','--numberOfFiles', type=int, default=None,  help="Number of files to process per sample")

    opt = parser.parse_args()

    print(opt)

    #from dask.distributed import Client
    import time
    
    #client = Client("tls://localhost:8786")
    #ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/NanoGEN/"
    ntuples_location = "/net/data_cms/institut_3a/NanoGEN/"
    p2016_DYn_250_400 = ntuples_location + "/DYnJetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Summer15/FromGridPack-12Aug2021/210812_100639/0000/"
    p2017_DY1_250_400 = ntuples_location + "/DY1JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-12Aug2021/210812_100210/0000/"
    p2017_DY2_250_400 = ntuples_location + "/DY2JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-12Aug2021/210812_100403/0000/"

    ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/NanoGEN/"
    p2016_DYn_100_250 = ntuples_location + "/DYnJetsToLL_LHEZpT_100-250_TuneCUET8M1_13TeV_Summer15/FromGridPack-19Oct2021/211019_115119/0000/"
    p2017_DY1_150_250 = ntuples_location + "/DY1JetsToLL_LHEZpT_150-250_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_114808/0000/"
    p2017_DY2_150_250 = ntuples_location + "/DY2JetsToLL_LHEZpT_150-250_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_115012/0000/"

    p2016_DYn_250_400 = ntuples_location + "/DYnJetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Summer15/FromGridPack-19Oct2021/211019_110125/0000/"
    p2017_DY1_250_400 = ntuples_location + "/DY1JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_110316/0000/"
    p2017_DY2_250_400 = ntuples_location + "/DY2JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_105906/0000/"


    file_list = {
        '2016_DYnJ' :  si.getRootFilesFromPath(p2016_DYn_100_250, opt.numberOfFiles),
        '2017_DY1J' :  si.getRootFilesFromPath(p2017_DY1_150_250, opt.numberOfFiles),
        '2017_DY2J' :  si.getRootFilesFromPath(p2017_DY2_150_250, opt.numberOfFiles),

        #'2016_DYnJ' :  si.getRootFilesFromPath(p2016_DYn_250_400, opt.numberOfFiles),
        #'2017_DY1J' :  si.getRootFilesFromPath(p2017_DY1_250_400, opt.numberOfFiles),
        #'2017_DY2J' :  si.getRootFilesFromPath(p2017_DY2_250_400, opt.numberOfFiles),

        #'2017_DY1J' :  [p2017_DY1_250_400+"/Tree_1.root"],
        #'2017_DY2J' :  [p2017_DY2_250_400+"/Tree_1.root"],
        #'2016_DYnJ' :  [p2016_DYn_250_400+"/Tree_1.root"],
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
            for key2, value2 in output['cutflow'][key].items():
                print(key, key2,value2)
        for key, value in output['sumw'].items():
            print(key, value)

        
        
if __name__ == "__main__":
    print("This is the __main__ part")

    import time
    start_time = time.time()
    main()
    finish_time = time.time()

    print("Total runtime in seconds: " + str(finish_time - start_time))
