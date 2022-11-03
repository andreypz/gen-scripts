#!/usr/bin/env python3

from os import listdir, makedirs, path, system, getpid
import psutil
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pickle as pkl
from matplotlib import pyplot as plt
import hist as Hist
from hist.intervals import ratio_uncertainty
import mplhep as hep
import coffea.processor as processor
import awkward as ak
from coffea.nanoevents import NanoEventsFactory
from functools import partial
#from coffea.nanoevents import NanoAODSchema
from Coffea_NanoGEN_schema import NanoGENSchema
import sampleInfo as si

scaleout=200

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

        axis = {
            #"dataset"     : Hist.axis.StrCategory([],      name="dataset", label="Primary dataset", growth=True),
            "LHE_Vpt"     : Hist.axis.Regular(100, 0, 400, name="LHE_Vpt", label="LHE V PT [GeV]"),
            "LHE_HT"      : Hist.axis.Regular(100, 0, 1000, name="LHE_HT", label="LHE HT [GeV]"),
            'wei'         : Hist.axis.Regular(100, -1000, 10000, name="wei", label="wei"),
            'wei_sign'    : Hist.axis.Regular(50, -2, 2,   name="wei",      label="wei"),
            'nlep'        : Hist.axis.Regular(12, 0, 6,    name="nlep",     label="nlep"),
            'lep_eta'     : Hist.axis.Regular(50, -5, 5,   name="lep_eta",  label="lep_eta"),
            'lep_pt'      : Hist.axis.Regular(50, 0, 500,  name="lep_pt",   label="lep_pt"),
            'dilep_m'     : Hist.axis.Regular(50, 50, 120, name="dilep_m",  label="dilep_m"),
            'dilep_pt'    : Hist.axis.Regular(100, 0, 600, name="dilep_pt", label="dilep_pt"),
            'njet25'      : Hist.axis.Regular(12, 0, 6,    name="njet25",   label="njet25"),
            'jet_eta'     : Hist.axis.Regular(50, -5, 5,   name="jet_eta",  label="jet_eta"),
            'jet_pt'      : Hist.axis.Regular(50, 0, 500,  name="jet_pt",   label="jet_pt"),
            'dijet_m'     : Hist.axis.Regular(50, 0, 1200, name="dijet_m",  label="dijet_m"),
            'dijet_pt'    : Hist.axis.Regular(100, 0, 600, name="dijet_pt", label="dijet_pt"),
            'dijet_dr'    : Hist.axis.Regular(50, 0, 5,    name="dijet_dr", label="dijet_dr"),
            #'dijet_dr_neg': Hist.axis.Regular(50, 0, 5,    name="dijet_dr", label="dijet_dr")
        }

        self._accumulator = processor.dict_accumulator(
            {observable : Hist.Hist(var_axis, name="Counts", storage="Weight") for observable, var_axis in axis.items() if observable!="dataset"}
        )
        self._accumulator['cutflow'] = processor.defaultdict_accumulator( partial(processor.defaultdict_accumulator, int) )
        self._accumulator["sumw"] =  0

        print("\t Init : ", psutil.Process(getpid()).memory_info().rss / 1024 ** 2, "MB")

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator
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
                             ak.fill_none( (np.abs(particles.parent.pdgId) != 15), True) &
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

        output["sumw"] += np.sum(weight_nosel)

        output['LHE_Vpt'].fill(LHE_Vpt=LHE_Vpt, weight=weight_nosel)
        output['LHE_HT'].fill(LHE_HT=LHE_HT, weight=weight_nosel)

        output['wei'].fill(wei=weight_nosel, weight=weight_nosel)
        output['wei_sign'].fill(wei=weight_nosel/np.abs(weight_nosel), weight=weight_nosel)

        output['nlep'].fill(nlep=ak.num(leptons), weight=weight_nosel)


        dileptons = ak.combinations(leptons, 2, fields=['i0', 'i1'])

        pt25  = ((dileptons['i0'].pt > 25) | (dileptons['i1'].pt > 25))
        Zmass_cut = (((dileptons['i0'] + dileptons['i1']).mass - 91.19) < 15)
        Vpt_cut = ( (dileptons['i0'] + dileptons['i1']).pt > 10 )
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


        #LHE_Njets_cut = (LHE_Njets>=0)
        selection_2l   = two_lep
        selection_2l2j = two_lep & two_jets & LHE_vpt_cut
        #full_selection = two_lep & two_jets & Vpt_cut
        #full_selection = two_lep & two_jets & LHE_vpt_cut & vmass_cut
        #full_selection = two_lep & two_jets & vpt_cut & vmass_cut


        events_2l   = events[selection_2l]
        events_2l2j = events[selection_2l2j]

        output['cutflow'][dataset]["events_2l"] += len(events_2l)
        output['cutflow'][dataset]["events_2l2j"] += len(events_2l2j)


        if dataset in ['DYJets_inc_FXFX','DYJets_MiNNLO_Mu_Supp']:
            weight_full = events_2l2j.genWeight
            weight_2l = events_2l.genWeight
        else:
            weight_full = np.sign(events_2l2j.genWeight)
            weight_2l = np.sign(events_2l.genWeight)
        #weight = np.ones(len(events_2l2j))
        weight2_full = np.repeat(np.array(weight_full),2)
        weight2_2l = np.repeat(np.array(weight_2l),2)
        #print("weight length:", len(weight), len(weight2))
        #print(leptons.eta[full_selection][:,0:2])


        output['njet25'].fill(njet25=ak.num(good_jets[selection_2l]), weight=weight_2l)

        dijets = good_jets[selection_2l2j]
        dijet = dijets[:, 0] + dijets[:, 1]

        dijet_pt = dijet.pt
        dijet_m  = dijet.mass
        dijet_dr = dijets[:, 0].delta_r(dijets[:, 1])


        output['dilep_m'].fill(dilep_m=ak.flatten(vmass[selection_2l2j]), weight=weight_full)
        output['dilep_pt'].fill(dilep_pt=ak.flatten(vpt[selection_2l2j]), weight=weight_full)

        output['lep_eta'].fill(lep_eta=ak.flatten(leptons.eta[selection_2l2j][:,0:2]), weight=weight2_full)
        output['lep_pt'].fill(lep_pt=ak.flatten(leptons.pt[selection_2l2j][:,0:2]), weight=weight2_full)

        output['jet_eta'].fill(jet_eta=ak.flatten(good_jets.eta[selection_2l2j][:,0:2]), weight=weight2_full)
        output['jet_pt'].fill(jet_pt=ak.flatten(good_jets.pt[selection_2l2j][:,0:2]), weight=weight2_full)

        output['dijet_dr'].fill(dijet_dr=dijet_dr, weight=weight_full)
        output['dijet_m'].fill(dijet_m=dijet_m, weight=weight_full)
        output['dijet_pt'].fill(dijet_pt=dijet_pt, weight=weight_full)

        ##print("Negative DRs:", dijet_dr[weight<0])
        ##print("Negative wei:", weight[weight<0])
        #neg_wei = np.abs(weight_full[weight_full<0])
        #neg_wei_dr = dijet_dr[weight_full<0]
        #output['dijet_dr_neg'].fill(dijet_dr=neg_wei_dr, weight=neg_wei)

        return {dataset:output}

    def postprocess(self, accumulator):
        lumi = 11 # random lumi, it does not matter here

        for dataset in accumulator:
            print(dataset, accumulator[dataset]['sumw'])
            #print(accumulator[dataset])


        group_axis = Hist.axis.StrCategory([], name="ds_scaled", label="Dataset merged", growth=True),

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
            weights = {sname : lumi*sampleInfo[sname]['xsec']*sampleInfo[sname]['kfac']/accumulator[sname]['sumw'] for sname in accumulator.keys()}
            if self.verblvl>0:
                print("weights = ", weights)

            for ds in accumulator:
                for key in accumulator[ds]:
                    if key not in ['cutflow','sumw']:
                        accumulator[ds][key] *= weights[ds]

            #accumulator["DYJets_MiNNLO"] = processor.accumulate([accumulator["DYJets_inc_MiNNLO_Mu"],accumulator["DYJets_inc_MiNNLO_El"]] )
            accumulator["DYJets_MiNNLO"] = accumulator["DYJets_inc_MiNNLO_Mu"]
            del accumulator["DYJets_inc_MiNNLO_Mu"]
            accumulator["DYJets_MiNNLO_Supp"] = accumulator["DYJets_MiNNLO_Mu_Supp"]
            del accumulator["DYJets_MiNNLO_Mu_Supp"]

            """
                        accumulator[ds][key] = accumulator[ds][key].group('dataset', group_axis, {'DYJets_inc_MLM':  ['DYJets_inc_MLM'],
                                                                                                  'DYJets_inc_FXFX': ['DYJets_inc_FXFX'],
                                                                                                  'DYJets_MiNNLO': ['DYJets_inc_MiNNLO_Mu','DYJets_inc_MiNNLO_El'],
                                                                                                  'DYJets_MiNNLO_Supp': ['DYJets_MiNNLO_Mu_Supp'],
                                                                                                  'DYJets_NJ_FXFX':  ['DYJets_0J','DYJets_1J','DYJets_2J'],
                                                                                                  'DYJets_PT_FXFX':  ['DYJets_Pt50To100','DYJets_Pt100To250','DYJets_Pt250To400','DYJets_Pt400To650','DYJets_Pt650ToInf'],
                                                                                                  'xDYJets_PT_FXFX':  ['xDYJets_Pt50To100','xDYJets_Pt100To250','xDYJets_Pt250To400','xDYJets_Pt400To650','xDYJets_Pt650ToInf'],
                                                                                                  'DYJets_HT_MLM': ['DYJets_HT70to100','DYJets_HT100to200','DYJets_HT200to400','DYJets_HT400to600','DYJets_HT600to800','DYJets_HT800to1200','DYJets_HT1200to2500','DYJets_HT2500toInf'],
                                                                                                  'DYJets_HERWIG':  ['DYJets_HERWIG'],
                                                                                                  
                                                                                              })
            """
        return accumulator


def plot(accumulated, opt, fromPickles=False):
    '''Plot all histograms'''
    if not path.exists(opt.outdir):
        makedirs(opt.outdir)

    if not fromPickles:
        pkl.dump( accumulated,  open(opt.outdir+'/Pickles.pkl',  'wb')  )

    datasets = ['DYJets_MiNNLO','DYJets_MiNNLO_Supp','DYJets_inc_FXFX']
    for dataset, accum in accumulated.items():
        if opt.debug>1:
            print(dataset, accum)
        observables = accumulated[dataset].keys()
        #datasets.append(dataset)

    if opt.debug>0:
        print("observables:", observables)
        print("datasets:", datasets)
        
    for observable in observables:
        if observable in ['cutflow','sumw']: continue
        if observable=="dijet_dr_neg":
            obs_axis="dijet_dr"
        elif observable=="wei_sign":
            obs_axis="wei"
        else:
            obs_axis=observable
        #if opt.debug>0:
        #    print (observable, type(accumulated[]))
        #if type(histogram) is hist.hist_tools.Hist:
        #    print(observable, "I am a Hist", histogram)
        #    if not histogram.values():
        #        print("This hist is empty!", histogram.values())
        #        continue
        #else:
        #    continue

        plt.gcf().clf()


        #print(histogram.axes())
        #print(list(map(lambda x:x.name, histogram.axes() )))
        print("Plotting dataset = ", datasets[0], "; Obs = ", observable)
        fig, (ax, rax) = plt.subplots(nrows=2, ncols=1, figsize=(8,8),
                                      gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
        fig.subplots_adjust(hspace=0.05, top=0.92, bottom=0.2, right=0.97)
        hep.cms.label("Preliminary", com="13", data=False, loc=0, ax=ax)

        if opt.debug>1:
            print(accumulated[datasets[0]][observable])

        for d in datasets:
            hep.histplot( accumulated[d][observable], label=d, histtype="step",yerr=True, ax=ax)

        ax.set_ylim(0, None)
        if obs_axis in ['LHE_HT','wei']:
            ax.set_ylim(1, None)
            ax.set_yscale('log')
            
        leg = ax.legend()

        #samp1=['DYJets_inc_MLM','MLM']
        #samp1=['DYJets_NJ_FXFX','NJ_FXFX']
        samp0=['DYJets_inc_FXFX','FXFX']
        samp1=['DYJets_MiNNLO','MiNNLO']
        samp2=['DYJets_MiNNLO_Supp','MiNNLO_Supp']
        #samp3=['DYJets_HERWIG','HERWIG']        
        #print(histogram["DYJets_inc_MLM"].axes())


        rax.set_xlabel('')

        rax.errorbar(x=accumulated[datasets[0]][observable].axes[0].centers,
                     y=accumulated[datasets[0]][observable].values()/accumulated[datasets[1]][observable].values(),
                     yerr=ratio_uncertainty(
                         accumulated[datasets[0]][observable].values(), accumulated[datasets[1]][observable].values()
                     ),
                     marker="o", linestyle="none", color='c', elinewidth=1,
                     label=samp0[1]+"/"+samp1[1],
        )

        rax.errorbar(x=accumulated[datasets[0]][observable].axes[0].centers,
                     y=accumulated[datasets[2]][observable].values()/accumulated[datasets[0]][observable].values(),
                     yerr=ratio_uncertainty(
                         accumulated[datasets[2]][observable].values(), accumulated[datasets[0]][observable].values()
                     ),
                     marker="v", linestyle="none", color='brown', elinewidth=1,
                     label=samp0[1]+"/"+samp2[1],
        )

        rax.errorbar(x=accumulated[datasets[1]][observable].axes[0].centers,
                     y=accumulated[datasets[1]][observable].values()/accumulated[datasets[2]][observable].values(),
                     yerr=ratio_uncertainty(
                         accumulated[datasets[1]][observable].values(), accumulated[datasets[2]][observable].values()
                     ),
                     marker=">", linestyle="none", color='m', elinewidth=1,
                     label=samp1[1]+"/"+samp2[1],
        )
            
        legrx = rax.legend(loc="upper center", ncol=3)
        rax.axhline(y=1.0, linestyle="dashed", color="gray")
        rax.set_ylabel('Ratios')
        rax.set_ylim(0.6,1.6)
        rax.set_xlabel(observable)
        
        plt.gcf().savefig(f"{opt.outdir}/{observable}.png", bbox_inches='tight')


def plotFromPickles(inputfile, opt):
    hists = pkl.load(open(inputfile,'rb'))
    plot(hists, opt, fromPickles=False)

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
                f'echo {getcwd()}',
                f'ls {getcwd()}',
                f'source {getcwd()}/CondaSetup.sh',
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
                            init_blocks=scaleout,
                            max_blocks=scaleout+10,
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




        plot(output, opt)


        for key, value in output.items():
            #print(key, value)
            for key2, value2 in output[key]['cutflow'][key].items():
                print(key, key2, value2)
        for key, value in output.items():
            print(key, value['sumw'])



if __name__ == "__main__":
    print("This is the __main__ part")

    import time
    start_time = time.time()
    main()
    finish_time = time.time()

    print("Total runtime in seconds: " + str(finish_time - start_time))
