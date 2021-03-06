#!/usr/bin/env python3

from os import listdir, makedirs, path, system
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pickle as pkl
from matplotlib import pyplot as plt
from coffea import hist
import coffea.processor as processor
import awkward as ak
from coffea.nanoevents import NanoEventsFactory
from functools import partial
#from coffea.nanoevents import NanoAODSchema
from Coffea_NanoGEN_schema import NanoGENSchema
import sampleInfo as si

def isClean(obj_A, obj_B, drmin=0.4):
    # From: https://github.com/oshadura/topcoffea/blob/master/topcoffea/modules/objects.py
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

class Processor(processor.ProcessorABC):
    def __init__(self, proc_type, verblvl):
        print("Process type:", proc_type)
        self.proc_type = proc_type
        self.verblvl = verblvl

        axis = { "dataset": hist.Cat("dataset", ""),
                 "LHE_Vpt": hist.Bin("LHE_Vpt", "LHE V PT [GeV]", 100, 0, 400),
                 "LHE_HT":  hist.Bin("LHE_HT", "LHE HT [GeV]", 100, 0, 1000),
                 'wei'         : hist.Bin("wei", "wei", 100, -1000, 10000),
                 'wei_sign'    : hist.Bin("wei", "wei", 50, -2, 2),
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
        LHE_HT = events.LHE['HT']
        #LHE_Njets = events.LHE['LHE_Njets'] # Does not exist in NanoV2
        #print(LHE_Vpt)
        # We can define a new key for cutflow (in this case 'all events').
        # Then we can put values into it. We need += because it's per-chunk (demonstrated below)
        output['cutflow'][dataset]['all_events'] += ak.size(LHE_Vpt)
        output['cutflow'][dataset]['number_of_chunks'] += 1

        particles = events.GenPart
        #leptons = particles[ (np.abs(particles.pdgId) == 13) & (particles.status == 1) & (np.abs(particles.eta)<2.5) ]
        leptons = particles[ ((np.abs(particles.pdgId) == 11) | (np.abs(particles.pdgId) == 13) ) & 
                             (particles.status == 1) & (np.abs(particles.eta)<2.5) & (particles.pt>20) ]
                
        genjets = events.GenJet
        jets25 = genjets[ (np.abs(genjets.eta) < 2.5)  &  (genjets.pt > 25) ]

        LHEP = events.LHEPart
        LHEjets  = LHEP[ ( (np.abs(LHEP.pdgId) == 1) | (np.abs(LHEP.pdgId) == 2) | (np.abs(LHEP.pdgId) == 3 ) |
                           (np.abs(LHEP.pdgId) == 4) | (np.abs(LHEP.pdgId) == 5) | (np.abs(LHEP.pdgId) == 21 ) ) &
                         (LHEP.status==1) ]
        LHE_Njets = ak.num(LHEjets)
        

        if dataset in ['DYJets_inc_FXFX','DYJets_MiNNLO_Mu_Supp']:
            weight_nosel = events.genWeight
        else:
            weight_nosel= np.sign(events.genWeight)

        if self.verblvl>0:
            print("\n",dataset, "wei:", weight_nosel)

        output["sumw"][dataset] += np.sum(weight_nosel)

        output['LHE_Vpt'].fill(dataset=dataset, LHE_Vpt=LHE_Vpt, weight=weight_nosel)
        output['LHE_HT'].fill(dataset=dataset, LHE_HT=LHE_HT, weight=weight_nosel)

        output['wei'].fill(dataset=dataset, wei=weight_nosel, weight=weight_nosel)
        output['wei_sign'].fill(dataset=dataset, wei=weight_nosel/np.abs(weight_nosel), weight=weight_nosel)

        output['nlep'].fill(dataset=dataset, nlep=ak.num(leptons), weight=weight_nosel)


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

        # jets25['isClean'] = isClean(jets25, leptons, drmin=0.5)
        #j_isclean = isClean(jets25, leptons, drmin=0.5)
        # From: https://github.com/CoffeaTeam/coffea/discussions/497#discussioncomment-600052
        j_isclean = ak.all(jets25.metric_table(leptons) > 0.5, axis=2)
        # NB: this gives identical result to the isClean() fuction above

        #good_jets = jets
        good_jets = jets25[j_isclean]
        two_jets = (ak.num(good_jets) >= 2)

        output['njet25'].fill(dataset=dataset, njet25=ak.num(good_jets), weight=weight_nosel)

        LHE_Njets_cut = (LHE_Njets>=0)
        full_selection = two_lep & two_jets & LHE_vpt_cut & LHE_Njets_cut
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


        if dataset in ['DYJets_inc_FXFX','DYJets_MiNNLO_Mu_Supp']:
            weight = selected_events.genWeight
        else:
            weight = np.sign(selected_events.genWeight)
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

        group_axis = hist.Cat('ds_scaled', 'ds_scaled')
        if self.proc_type=="pre":
            #xs = si.xs_150_250
            xs = si.xs_250_400
            print("Cross sections for normalization:", xs)

            weights = { '2016_DYnJ': lumi*xs['2016_DYnJ']/accumulator['sumw']['2016_DYnJ'],
                        '2017_DY1J': lumi*xs['2017_DY1J']/accumulator['sumw']['2017_DY1J'],
                        '2017_DY2J': lumi*xs['2017_DY2J']/accumulator['sumw']['2017_DY2J'],
                    }
            if self.verblvl>0:
                print("weights = ", weights)
                
            for key in accumulator:
                if key not in ['cutflow','sumw']:
                    accumulator[key].scale(weights, axis='dataset')

                    accumulator[key] = accumulator[key].group('dataset', group_axis, {'2016_DY 1+2j': ['2016_DYnJ'],
                                                                                      '2017_DY 1+2j': ['2017_DY1J', '2017_DY2J'],
                                                                                  })
        elif self.proc_type=="ul":
            sampleInfo = si.ReadSampleInfoFile('mc_vjets_samples.info')            
            weights = {sname : lumi*sampleInfo[sname]['xsec']*sampleInfo[sname]['kfac']/accumulator['sumw'][sname] for sname in accumulator['sumw'].keys()}
            if self.verblvl>0:
                print("weights = ", weights)
            
            for key in accumulator:
                if key not in ['cutflow','sumw']:
                    accumulator[key].scale(weights, axis='dataset')
                    accumulator[key] = accumulator[key].group('dataset', group_axis, {'DYJets_inc_MLM':  ['DYJets_inc_MLM'],
                                                                                      'DYJets_inc_FXFX': ['DYJets_inc_FXFX'],
                                                                                      'DYJets_MiNNLO': ['DYJets_inc_MiNNLO_Mu','DYJets_inc_MiNNLO_El'],
                                                                                      'DYJets_MiNNLO_Supp': ['DYJets_MiNNLO_Mu_Supp'],
                                                                                      'DYJets_NJ_FXFX':  ['DYJets_0J','DYJets_1J','DYJets_2J'],
                                                                                      'DYJets_PT_FXFX':  ['DYJets_Pt50To100','DYJets_Pt100To250','DYJets_Pt250To400','DYJets_Pt400To650','DYJets_Pt650ToInf'],
                                                                                      'xDYJets_PT_FXFX':  ['xDYJets_Pt50To100','xDYJets_Pt100To250','xDYJets_Pt250To400','xDYJets_Pt400To650','xDYJets_Pt650ToInf'],
                                                                                      'DYJets_HT_MLM': ['DYJets_HT70to100','DYJets_HT100to200','DYJets_HT200to400','DYJets_HT400to600','DYJets_HT600to800','DYJets_HT800to1200','DYJets_HT1200to2500','DYJets_HT2500toInf'],
                                                                                      'DYJets_HERWIG':  ['DYJets_HERWIG'],

                                                                                  })
                    
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

def plot(histograms, outdir, proc_type, fromPickles=False):
    '''Plots all histograms. No need to change.'''
    if not path.exists(outdir):
        makedirs(outdir)

    if not fromPickles:
        pkl.dump( histograms,  open(outdir+'/Pickles.pkl',  'wb')  )

    for observable, histogram in histograms.items():
        if observable=="dijet_dr_neg":
            obs_axis="dijet_dr"
        elif observable=="wei_sign":
            obs_axis="wei"
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
        if 'ds_scaled' in axes:
            if proc_type=="ul":
                print("Plotting for UL", "axis = ", obs_axis)
                fig, (ax, rax) = plt.subplots(nrows=2, ncols=1, figsize=(7,7),
                                              gridspec_kw={"height_ratios": (2, 1)},sharex=True)
                fig.subplots_adjust(hspace=.07)
                
                hist.plot1d(histogram, overlay='ds_scaled', ax=ax, line_opts={"color":['C2','C1','C0']}, overflow='none')
                ax.set_ylim(0, None)
                if obs_axis in ['LHE_HT','wei']:
                    ax.set_ylim(1, None)
                    ax.set_yscale('log')

                leg = ax.legend()

                #samp1=['DYJets_inc_MLM','MLM']
                #samp1=['DYJets_NJ_FXFX','NJ_FXFX']
                samp1=['DYJets_inc_FXFX','FXFX']
                samp2=['DYJets_MiNNLO','MiNNLO']
                samp3=['DYJets_MiNNLO_Supp','MiNNLO_Supp']
                #samp3=['DYJets_HERWIG','HERWIG']

                #print(histogram["DYJets_inc_MLM"].axes())

                r1 = hist.plotratio(num = histogram[samp1[0]].project(obs_axis),
                                    denom = histogram[samp2[0]].project(obs_axis),
                                    error_opts={'color': 'c', 'marker': 'o'},
                                    ax=rax,
                                    denom_fill_opts={},
                                    guide_opts={},
                                    unc='num',
                                    label=samp1[1]+"/"+samp2[1]
                                )
                
                hist.plotratio(num = histogram[samp1[0]].project(obs_axis),
                               denom = histogram[samp3[0]].project(obs_axis),
                               error_opts={'color': 'brown', 'marker': 'v'},
                               ax=rax,
                               clear = False,
                               label=samp1[1]+"/"+samp3[1],
                               unc='num'
                           )

                hist.plotratio(num = histogram[samp2[0]].project(obs_axis),
                               denom = histogram[samp3[0]].project(obs_axis),
                               error_opts={'color': 'm', 'marker': '>'},
                               ax=rax,
                               clear = False,
                               label=samp2[1]+"/"+samp3[1],
                               unc='num'
                           )
                legrx = rax.legend(loc="upper center", ncol=3)

                rax.set_ylabel('Ratios')
                rax.set_ylim(0.6,1.6)
                

            else:
                
                #hist.plot1d(histogram, overlay='dataset', line_opts={}, overflow='none')
                #plt.gca().autoscale()
                
                
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
                
        else:
            print("axes= ", axes)
            print("This should not happen. I'm not sure what to do.")

        plt.gcf().savefig(f"{outdir}/{observable}.png", bbox_inches='tight')

    #fracOfNegWeiPlot(histograms, outdir, "2016")
    #fracOfNegWeiPlot(histograms, outdir, "2017")

def plotFromPickles(inputfile, outdir, proc_type):
    hists = pkl.load(open(inputfile,'rb'))
    plot(hists, outdir, proc_type, fromPickles=False)

def retry_handler(exception, task_record):
    from parsl.executors.high_throughput.interchange import ManagerLost
    if isinstance(exception, ManagerLost):
            return 0.1
    else:
        return 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run quick plots from NanoGEN input files')
    #parser.add_argument("inputfile")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument('--pkl', type=str, default=None,  help="Make plots from pickled file.")
    parser.add_argument('-n','--numberOfFiles', type=int, default=None,  help="Number of files to process per sample")
    parser.add_argument('-t','--proc_type', type=str, default="ul", choices=["ul","pre"], help="Version of the code to run. 'ul' -- for UL samples; 'pre' - pre-UP samples (2016/2017 study)")
    parser.add_argument('-e','--executor', type=str, default="local", choices=["local","dask","parsl"], help="Executor")
    parser.add_argument("-d","--debug",  type=int,  default=0, help="Verbose level for debugging")


    opt = parser.parse_args()

    print(opt)

    import time


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

        #ntuples_location = "/net/data_cms/institut_3a/NanoAOD/"
        #p2016_DYn_250_400 = ntuples_location + "Test_ZH_HToCC_ZToNuNu_AK15"
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
    elif opt.proc_type=="ul" and opt.pkl==None:
        pkl_file = "./VJetsPickle.pkl"
        xroot = 'root://grid-cms-xrootd.physik.rwth-aachen.de/'
        #xroot = 'root://xrootd-cms.infn.it/'
        sampleInfo = si.ReadSampleInfoFile('mc_vjets_samples.info')

        file_list = {
            sname: si.makeListOfInputRootFilesForProcess(sname, sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles, checkOpen=False) for sname in sampleInfo
        }
        
        #file_list['DYJets_MiNNLO_Mu_Supp'] = si.makeListOfInputRootFilesForProcess("DYJets_MiNNLO_Mu_Supp", sampleInfo, pkl_file, xroot, lim=20, checkOpen=True)
        #file_list = {'DYJets_HERWIG': [#'~/work/DYToLL_NLO_5FS_TuneCH3_13TeV_matchbox_herwig7_cff_py_GEN_NANOGEN.root',
        #                               '~/work/DYToLL_NLO_5FS_TuneCH3_13TeV_matchbox_herwig7_cff_py_GEN_NANOGEN_inNANOAODGEN.root']}
        
        '''
        file_list = {
            'DYJets_inc_MLM': si.makeListOfInputRootFilesForProcess("DYJets_inc_MLM", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_inc_FXFX': si.makeListOfInputRootFilesForProcess("DYJets_inc_FXFX", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_inc_MiNNLO_Mu': si.makeListOfInputRootFilesForProcess("DYJets_inc_MiNNLO_Mu", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_inc_MiNNLO_El': si.makeListOfInputRootFilesForProcess("DYJets_inc_MiNNLO_El", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            
            'DYJets_0J': si.makeListOfInputRootFilesForProcess("DYJets_0J", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_1J': si.makeListOfInputRootFilesForProcess("DYJets_1J", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            'DYJets_2J': si.makeListOfInputRootFilesForProcess("DYJets_2J", sampleInfo, pkl_file, xroot, lim=opt.numberOfFiles),
            #'DYJets_inc_MLM': ['/user/andreypz/ZH_HCC_ZLL_NanoV6_2017_7C7E.root']
        }
        '''
        print(file_list.keys())

    if opt.pkl!=None:
        plotFromPickles(opt.pkl, opt.outdir, opt.proc_type)
    else:
        if opt.executor=="dask":
            #from dask_jobqueue.htcondor import HTCondorCluster
            from dask_jobqueue import HTCondorCluster
            cluster = HTCondorCluster(cores=24, memory="4GB", disk="4GB")
            cluster.scale(jobs=10)  # ask for 10 jobs
            
            from dask.distributed import Client
            #client = Client(n_workers=4, threads_per_worker=2)
            client = Client(cluster)
            print(client)

            output = processor.run_uproot_job(file_list,
                                              treename = 'Events',
                                              processor_instance = Processor(opt.proc_type, verblvl=opt.debug),
                                              executor = processor.dask_executor,
                                              executor_args = {'client': client, 'schema': NanoAODSchema}
            )

        elif  opt.executor=="parsl":
           

            try:
                from os import popen, environ, getcwd
                _x509_localpath = [l for l in popen('voms-proxy-info').read().split("\n") if l.startswith('path')][0].split(":")[-1].strip()
            except:
                raise RuntimeError("x509 proxy could not be parsed, try creating it with 'voms-proxy-init'")

            print(_x509_localpath)
            _x509_path = environ['HOME'] + f'/.{_x509_localpath.split("/")[-1]}'
            system(f'cp {_x509_localpath} {_x509_path}')
            
            env_extra = [
                'export XRD_RUNFORKHANDLER=1',
                f'export X509_USER_PROXY={_x509_path}',
                f'export X509_CERT_DIR={environ["X509_CERT_DIR"]}',
                f"export PYTHONPATH=$PYTHONPATH:{getcwd()}",
            ]
            condor_extra = [
                'source ~/work/vjets/conda_setup.sh',
                'conda activate coffea37',
                'echo LETSGO'
            ]
            
            import parsl

            from parsl.config import Config
            from parsl.executors import HighThroughputExecutor
            from parsl.providers import CondorProvider
            from parsl.addresses import address_by_hostname, address_by_query 
            
            # For local executor
            #from parsl.app.app import python_app, bash_app
            #from parsl.configs.local_threads import config
            #parsl.load(config)
            
            htex_config = Config(
                executors=[
                    HighThroughputExecutor(
                        label='coffea_parsl_condor',
                        address=address_by_query(),
                        max_workers=1,
                        provider=CondorProvider(
                            nodes_per_block=1,
                            init_blocks=20,
                            max_blocks=600,
                            scheduler_options='should_transfer_files = YES\n transfer_output_files = ""\n',
                            worker_init="\n".join(env_extra + condor_extra),
                            walltime="00:50:00",
                        ),
                    )
                ],
                retries=20,
                retry_handler=retry_handler,
            )
            dfk = parsl.load(htex_config)

            output = processor.run_uproot_job(file_list,
                                              treename = 'Events',
                                              processor_instance = Processor(opt.proc_type, verblvl=opt.debug),
                                              executor = processor.parsl_executor,
                                              executor_args = {
                                                  'skipbadfiles': True,
                                                  'schema': NanoGENSchema,
                                                  'config': None
                                              }
            )

        else:
            output = processor.run_uproot_job(file_list,
                                              treename = 'Events',
                                              processor_instance = Processor(opt.proc_type, verblvl=opt.debug),
                                              #executor = processor.iterative_executor,
                                              executor = processor.futures_executor,
                                              executor_args = {'schema': NanoGENSchema, "workers":10}
            )
            



        plot(output, opt.outdir, opt.proc_type)


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
