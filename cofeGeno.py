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
    def __init__(self, proc_type):
        print("Process type:", proc_type)
        self.proc_type = proc_type

        axis = { "dataset": hist.Cat("dataset", ""),
                 "LHE_Vpt": hist.Bin("LHE_Vpt", "LHE V PT [GeV]", 100, 0, 400),                 
                 'wei'         : hist.Bin("wei", "wei", 50, -10, 10), 
                 'nlep'        : hist.Bin("nlep", "nlep", 12, 0, 6), 
                 'lep_eta'     : hist.Bin("lep_eta", "lep_eta", 50, -5, 5), 
                 'lep_pt'      : hist.Bin("lep_pt", "lep_pt", 50, 0, 500), 
                 'dilep_m'     : hist.Bin("dilep_m", "dilep_m", 50, 50, 120), 
                 'dilep_pt'    : hist.Bin("dilep_pt", "dilep_pt", 100, 0, 600), 
                 'njet25'      : hist.Bin("njet25", "njet25", 12, 0, 6), 
                 'jet_eta'     : hist.Bin("jet_eta", "jet_eta", 50, -5, 5), 
                 'jet_pt'      : hist.Bin("jet_pt", "jet_pt", 50, 0, 500), 
                 'dijet_m'     : hist.Bin("dijet_m", "dijet_m", 50, 0, 1200), 
                 'dijet_pt'    : hist.Bin("dijet_pt", "dijet_pt", 100, 0, 600),
                 'dijet_dr'    : hist.Bin("dijet_dr", "dijet_dr", 50, 0, 5), 
                 'dijet_dr_neg': hist.Bin("dijet_dr", "dijet_dr", 50, 0, 5) 
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
        leptons = particles[ (np.abs(particles.pdgId) == 13) & (particles.status == 1) & (np.abs(particles.eta)<2.5) ]
        #leptons = particles[ ((np.abs(particles.pdgId) == 11) | (np.abs(particles.pdgId) == 13) ) &
        #                     (particles.status == 1) & (particles.pt>15) & (np.abs(particles.eta)<2.5) ]
        
        genjets = events.GenJet
        jets25 = genjets[ (np.abs(genjets.eta) < 2.5)  &  (genjets.pt > 25) ]

        #jets25  = particles[ ( (np.abs(particles.pdgId) == 1) | (np.abs(particles.pdgId) == 2) | (np.abs(particles.pdgId) == 3 ) |
        #                       (np.abs(particles.pdgId) == 4) | (np.abs(particles.pdgId) == 5) | (np.abs(particles.pdgId) == 21 ) ) &
        #                     (particles.status==1) & (particles.pt > 25) ]


        
        weight_nosel = events.genWeight
        output['LHE_Vpt'].fill(dataset=dataset, LHE_Vpt=LHE_Vpt, weight=weight_nosel)
        
        output["sumw"][dataset] += np.sum(weight_nosel)
        print(dataset, "wei:", weight_nosel)
        
        output['wei'].fill(dataset=dataset, wei=weight_nosel/np.abs(weight_nosel))
        
        
        output['nlep'].fill(dataset=dataset, nlep=ak.num(leptons))


        dileptons = ak.combinations(leptons, 2, fields=['i0', 'i1'])

        pt25  = ((dileptons['i0'].pt > 25) | (dileptons['i1'].pt > 25))
        Zmass_cut = (((dileptons['i0'] + dileptons['i1']).mass - 91.19) < 15)
        Vpt_cut = ( (dileptons['i0'] + dileptons['i1']).pt > 100 )
        dileptonMask = pt25 & Zmass_cut & Vpt_cut
        good_dileptons = dileptons[dileptonMask]
        
        vpt = (good_dileptons['i0'] + good_dileptons['i1']).pt
        vmass = (good_dileptons['i0'] + good_dileptons['i1']).mass

        two_lep = ak.num(good_dileptons) == 1
        

        if self.proc_type=="pre":
            #LHE_vpt_cut = (LHE_Vpt>=155) & (LHE_Vpt<=245)
            LHE_vpt_cut = (LHE_Vpt>=255) & (LHE_Vpt<=395)
        elif self.proc_type=="ul":
            LHE_vpt_cut = True

        jets25['isClean'] = isClean(jets25, leptons, drmin=0.5)
        j_isclean = isClean(jets25, leptons, drmin=0.5)

        #good_jets = jets
        good_jets = jets25[j_isclean]
        two_jets = (ak.num(good_jets) >= 2)
        
        output['njet25'].fill(dataset=dataset, njet25=ak.num(good_jets))

        full_selection = two_lep & two_jets & LHE_vpt_cut
        #full_selection = two_lep & two_jets & Vpt_cut
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
        weight2 = np.repeat(np.array(weight),2)
        #print("weight length:", len(weight), len(weight2))
        #print(leptons.eta[full_selection][:,0:2])
                        
        output['dilep_m'].fill(dataset=dataset, dilep_m=ak.flatten(vmass[full_selection]), weight=weight)
        output['dilep_pt'].fill(dataset=dataset, dilep_pt=ak.flatten(vpt[full_selection]), weight=weight)
        
        output['lep_eta'].fill(dataset=dataset, lep_eta=ak.flatten(leptons.eta[full_selection][:,0:2]), weight=weight2)
        output['lep_pt'].fill(dataset=dataset, lep_pt=ak.flatten(leptons.pt[full_selection][:,0:2]), weight=weight2)
        
        output['jet_eta'].fill(dataset=dataset, jet_eta=ak.flatten(good_jets.eta[full_selection][:,0:2]), weight=weight2)
        output['jet_pt'].fill(dataset=dataset, jet_pt=ak.flatten(good_jets.pt[full_selection][:,0:2]), weight=weight2)
        
        output['dijet_dr'].fill(dataset=dataset, dijet_dr=dijet_dr, weight=weight)
        output['dijet_m'].fill(dataset=dataset, dijet_m=dijet_m, weight=weight)
        output['dijet_pt'].fill(dataset=dataset, dijet_pt=dijet_pt, weight=weight)

        #print("Negative DRs:", dijet_dr[weight<0])
        #print("Negative wei:", weight[weight<0])
        neg_wei = np.abs(weight[weight<0])
        neg_wei_dr = dijet_dr[weight<0]
        output['dijet_dr_neg'].fill(dataset=dataset, dijet_dr=neg_wei_dr, weight=neg_wei)

        return output

    def postprocess(self, accumulator):

        lumi = 11 # random lumi, it does not matter here
        
        print(accumulator['sumw'])

        if self.proc_type=="pre":
            #xs = si.xs_150_250
            xs = si.xs_250_400
            print("Cross sections for normalization:", xs)
            
            weights = { '2016_DYnJ': lumi*xs['2016_DYnJ']/accumulator['sumw']['2016_DYnJ'],
                        '2017_DY1J': lumi*xs['2017_DY1J']/accumulator['sumw']['2017_DY1J'],
                        '2017_DY2J': lumi*xs['2017_DY2J']/accumulator['sumw']['2017_DY2J'],
                    }
            print(weights)

            scaled = hist.Cat('ds_scaled', 'ds_scaled')
            for key in accumulator:
                if key not in ['cutflow','sumw']:
                    accumulator[key].scale(weights, axis='dataset')
                    
                    accumulator[key] = accumulator[key].group('dataset', scaled, {'2016_DY 1+2j': ['2016_DYnJ'], 
                                                                                  '2017_DY 1+2j': ['2017_DY1J', '2017_DY2J'],
                                                                              })
        elif self.proc_type=="ul":

            xs = {"DYJets_inc_MLM": 1.6*5.35e+03,
                  "DYJets_inc_FXFX": 6.43e+03,
                  "DYJets_inc_MinNLO": 1.976e+03}
            
            weights = {"DYJets_inc_MLM":    lumi*xs['DYJets_inc_MLM']/accumulator['sumw']['DYJets_inc_MLM'],
                       "DYJets_inc_FXFX":   lumi*xs['DYJets_inc_FXFX']/accumulator['sumw']['DYJets_inc_FXFX'],
                       "DYJets_inc_MinNLO": lumi*xs['DYJets_inc_MinNLO']/accumulator['sumw']['DYJets_inc_MinNLO']}
            
            for key in accumulator:
                if key not in ['cutflow','sumw']:
                    accumulator[key].scale(weights, axis='dataset')

        return accumulator



def fracOfNegWeiPlot(histograms, outdir, year="2016"):

    if histograms["dijet_dr"] and histograms["dijet_dr_neg"]:
 
        print(histograms)
        h_tot = histograms["dijet_dr"]
        h_neg = histograms["dijet_dr_neg"]
        fig, ax = plt.subplots()
        
        #leg = plt.legend()
        hist.plotratio(num = h_neg[year+"_DY 1+2j"].project("dijet_dr"),
                       denom = h_tot[year+"_DY 1+2j"].project("dijet_dr"),
                       error_opts={'color': 'k', 'marker': '.'},
                       ax=ax,
                       #denom_fill_opts={},
                       #guide_opts={},
                       unc='num'
        )
        
        ax.set_ylabel('Negative/Total Ratio')
        ax.set_ylim(0.1,1.5)
        plt.title(f"Contribution from negative weights in {year} sample.")
        plt.gcf().savefig(f"{outdir}/NegWeiFrac_{year}.png", bbox_inches='tight')
    else:
        print("The hists for Neg weight plot do not exist!")
        
def plot(histograms, outdir, fromPickles=False):
    '''Plots all histograms. No need to change.'''
    if not path.exists(outdir):
        makedirs(outdir)

    if not fromPickles:
        pkl.dump( histograms,  open(outdir+'/Pickles.pkl',  'wb')  )

    for observable, histogram in histograms.items():
        if observable=="dijet_dr_neg": 
            obs_axis="dijet_dr"
        else:
            obs_axis=observable
        #print (observable, histogram, type(histogram))
        if type(histogram) is hist.hist_tools.Hist:
            print(observable, "I am a Hist", histogram)
            if not histogram.values():
                print("This hist is empty!", histogram.values())
                continue
        else:
            continue

        plt.gcf().clf()

        #print(histogram.axes())
        #print(list(map(lambda x:x.name, histogram.axes() )))
        axes = list(map(lambda x:x.name, histogram.axes() ))
        if 'dataset' in axes:
            hist.plot1d(histogram, overlay='dataset', line_opts={}, overflow='none')
            plt.gca().autoscale()

            #if opt.proc_type=="ul"

        elif 'ds_scaled' in axes:
            fig, (ax, rax) = plt.subplots(nrows=2, ncols=1, figsize=(7,7),
                                          gridspec_kw={"height_ratios": (3, 1)},sharex=True)
            fig.subplots_adjust(hspace=.07)
            
            hist.plot1d(histogram, overlay='ds_scaled', ax=ax, line_opts={}, overflow='none')
            ax.set_ylim(0, None)
            
            leg = ax.legend()
            print(histogram["2016_DY 1+2j"].axes())
            hist.plotratio(num = histogram["2017_DY 1+2j"].project(obs_axis),
                           denom = histogram["2016_DY 1+2j"].project(obs_axis),
                           error_opts={'color': 'k', 'marker': '.'},
                           ax=rax,
                           denom_fill_opts={},
                           guide_opts={},
                           unc='num'
                       )

            rax.set_ylabel('2017/2016 Ratio')
            rax.set_ylim(0.5,1.5)

        plt.gcf().savefig(f"{outdir}/{observable}.png", bbox_inches='tight')
            
    #fracOfNegWeiPlot(histograms, outdir, "2016")
    #fracOfNegWeiPlot(histograms, outdir, "2017")

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
    parser.add_argument('-t','--proc_type', type=str, default="ul", choices=["ul","pre"], help="Version of the code to run. 'ul' -- for UL samples; 'pre' - pre-UP samples (2016/2017 stadu)")

    opt = parser.parse_args()

    print(opt)

    #from dask.distributed import Client
    import time
    
    #client = Client("tls://localhost:8786")

    if opt.proc_type=="pre":
        #ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/NanoGEN/"
        ntuples_location = "/net/data_cms/institut_3a/NanoGEN/"
        p2016_DYn_250_400 = ntuples_location + "/DYnJetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Summer15/FromGridPack-12Aug2021/210812_100639/0000/"
        p2017_DY1_250_400 = ntuples_location + "/DY1JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-12Aug2021/210812_100210/0000/"
        p2017_DY2_250_400 = ntuples_location + "/DY2JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-12Aug2021/210812_100403/0000/"
        
        ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/NanoGEN/"
        p2016_DYn_100_250 = ntuples_location + "/DYnJetsToLL_LHEZpT_100-250_TuneCUET8M1_13TeV_Summer15/FromGridPack-19Oct2021/211019_115119/0000/"
        p2017_DY1_150_250 = ntuples_location + "/DY1JetsToLL_LHEZpT_150-250_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_114808/0000/"
        p2017_DY2_150_250 = ntuples_location + "/DY2JetsToLL_LHEZpT_150-250_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_115012/0000/"
        
        #p2016_DYn_250_400 = ntuples_location + "/DYnJetsToLL_LHEZpT_250-400_TuneCUET8M1_13TeV_Summer15/FromGridPack-19Oct2021/211019_110125/0000/"
        p2016_DYn_250_400 = ntuples_location + "/DYnJetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Summer15/FromGridPack-02Nov2021/211102_143539/0000/"
        p2017_DY1_250_400 = ntuples_location + "/DY1JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_110316/0000/"
        p2017_DY2_250_400 = ntuples_location + "/DY2JetsToLL_LHEZpT_250-400_TuneCP5_13TeV_Fall17/FromGridPack-19Oct2021/211019_105906/0000/"
        
        #ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/mc/"
        #p2016_DYn_250_400 = ntuples_location + "/RunIISummer16NanoAODv7/DYJetsToLL_Pt-250To400_TuneCUETP8M1_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/PUMoriond17_Nano02Apr2020_102X_mcRun2_asymptotic_v8-v1"
        #p2017_DY1_250_400 = ntuples_location + "/RunIIFall17NanoAODv7/DY1JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/"
        #p2017_DY2_250_400 = ntuples_location + "/RunIIFall17NanoAODv7/DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/"
        
        file_list = {
            #'2016_DYnJ' :  si.getRootFilesFromPath(p2016_DYn_100_250, opt.numberOfFiles),
            #'2017_DY1J' :  si.getRootFilesFromPath(p2017_DY1_150_250, opt.numberOfFiles),
            #'2017_DY2J' :  si.getRootFilesFromPath(p2017_DY2_150_250, opt.numberOfFiles),
            
            '2016_DYnJ' :  si.getRootFilesFromPath(p2016_DYn_250_400, opt.numberOfFiles),
            '2017_DY1J' :  si.getRootFilesFromPath(p2017_DY1_250_400, opt.numberOfFiles),
            '2017_DY2J' :  si.getRootFilesFromPath(p2017_DY2_250_400, opt.numberOfFiles),
            #'2017_DY1J' :  [p2017_DY1_250_400+"/Tree_1.root"],
            #'2017_DY2J' :  [p2017_DY2_250_400+"/Tree_1.root"],
            #'2016_DYnJ' :  [p2016_DYn_250_400+"/Tree_1.root"],
        }        
    elif opt.proc_type=="ul":
        pkl_file = "./VJetsPickle.pkl"
        xroot = 'root://grid-cms-xrootd.physik.rwth-aachen.de/'
        sampleInfo = si.ReadSampleInfoFile('mc_vjets_samples.info')
        
        file_list = {            
            'DYJets_inc_MLM': si.makeListOfInputRootFilesForProcess("DYJets_inc_MLM", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_inc_FXFX': si.makeListOfInputRootFilesForProcess("DYJets_inc_FXFX", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_inc_MinNLO': si.makeListOfInputRootFilesForProcess("DYJets_inc_MinNLO", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            
        }

    
    
    if opt.pkl!=None:
        plotFromPickles(opt.pkl, opt.outdir)
    else:
        output = processor.run_uproot_job(file_list,
                                          treename = 'Events',
                                          processor_instance = Processor(opt.proc_type),
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
