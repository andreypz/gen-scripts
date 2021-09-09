#!/usr/bin/env python3
from os import listdir, makedirs, path, system
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from coffea import hist
import coffea.processor as processor
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from Coffea_NanoGEN_schema import NanoGENSchema
from Coffea_NanoAOD_PP_schema import NanoAODPPSchema

from functools import partial

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

    #print(rootfiles)
    return rootfiles

def isClean(obj_A, obj_B, drmin=0.4):
    # From: https://github.com/oshadura/topcoffea/blob/master/topcoffea/modules/objects.py
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)

class Processor(processor.ProcessorABC):
    def __init__(self):

        axis = { "dataset": hist.Cat("dataset", "dataset"),
                 "PDFwei": hist.Cat("PDFwei", "PDF name"),
                 "LHE_Vpt": hist.Bin("LHE_Vpt", "V PT [GeV]", 100, 0, 600),
                 'wei'        : hist.Bin("wei", "wei", 50, -10, 10),
                 'nlep'       : hist.Bin("nlep", "nlep", 12, 0, 6),
                 'dilep_m'    : hist.Bin("dilep_m", "dilep_m", 50, 50, 120),
                 'dilep_pt'   : hist.Bin("dilep_pt", "dilep_pt", 100, 0, 600),
                 'njet15'     : hist.Bin("njet15", "njet15", 12, 0, 6),
                 'dijet_dr'   : hist.Bin("dijet_dr", "dijet_dr", 50, 0, 5),
                 'dijet_m'    : hist.Bin("dijet_m", "dijet_m", 50, 0, 1200),
                 'dijet_pt'   : hist.Bin("dijet_pt", "dijet_pt", 100, 0, 600)
             }

        self._accumulator = processor.dict_accumulator(
            {observable : hist.Hist("Counts", axis["dataset"], var_axis) for observable, var_axis in axis.items() if observable not in ["dataset", "PDFwei", "dilep_pt"]}
        )
        self._accumulator['dilep_pt'] = hist.Hist("Counts", axis["dataset"], axis["PDFwei"], axis["dilep_pt"])
        self._accumulator['cutflow'] = processor.defaultdict_accumulator( partial(processor.defaultdict_accumulator, int) )
        self._accumulator['sumw'] =  processor.defaultdict_accumulator( float )


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        #print(output)

        dataset = events.metadata["dataset"]

        #events["Factor2"] = 
        print("PDF LHE weights:", len(events.LHEPdfWeight), events.LHEPdfWeight)
        print(ak.num(events.LHEPdfWeight))

        LHE_Vpt = events.LHE['Vpt']
        output['cutflow'][dataset]['all_events'] += ak.size(LHE_Vpt)
        output['cutflow'][dataset]['number_of_chunks'] += 1

        #print(LHE_Vpt)

        weight_nosel = events.genWeight
        output["sumw"][dataset] += np.sum(weight_nosel)
        #print(weight_nosel)

        output['LHE_Vpt'].fill(dataset=dataset, LHE_Vpt=LHE_Vpt, weight=weight_nosel)

        output['wei'].fill(dataset=dataset, wei=weight_nosel/np.abs(weight_nosel))

        muons = events.Muon

        goodmuon = (
            (muons.pt > 15)
            & (abs(muons.eta) < 2.4)
            & (muons.pfRelIso04_all < 0.25)
            & (muons.looseId)
            & (np.abs(muons.dxy) < 0.05)
            & (np.abs(muons.dz) < 0.1)
        )
        nmuons = ak.sum(goodmuon, axis=1)

        lead_muon_pt = ak.firsts(muons[goodmuon]).pt > 20

        muons = muons[goodmuon]

        dimuons = ak.combinations(muons, 2, fields=['i0', 'i1'])
        opposites = (dimuons['i0'].charge != dimuons['i1'].charge)

        limits = ((dimuons['i0'] + dimuons['i1']).mass >= 60) & ((dimuons['i0'] + dimuons['i1']).mass < 120)

        good_dimuons = dimuons[opposites & limits]


        electrons = events.Electron
        abs_eta = np.abs(electrons.eta)
        goodelectron = (
            (electrons.pt > 15)
            & (abs_eta < 2.5)
            & (abs(electrons.dxy) < 0.05)
            & (abs(electrons.dz) < 0.1)
            & (electrons.lostHits < 2)
            & (electrons.miniPFRelIso_all < 0.4)
            & (((electrons.mvaFall17V2noIso > 0) & (abs_eta < 1.479)) | ((electrons.mvaFall17V2noIso > 0.7) & (abs_eta > 1.479) & (abs_eta < 2.5)))
        )
        electrons = electrons[goodelectron]

        vpt = (good_dimuons['i0'] + good_dimuons['i1']).pt
        vmass = (good_dimuons['i0'] + good_dimuons['i1']).mass

        output['nlep'].fill(dataset=dataset, nlep=nmuons)

        two_lep = ak.num(good_dimuons) == 1

        #print(good_dimuons[two_lep])
        #print(vmass[two_lep])
        
        MET = events.MET.pt

        jets = events.Jet
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 2.5)
            & jets.isTight
        ]

        jets['isClean'] = isClean(jets, electrons, drmin=0.4)& isClean(jets, muons, drmin=0.4)
        j_isclean = isClean(jets, electrons, drmin=0.4)& isClean(jets, muons, drmin=0.4)

        #good_jets = jets
        good_jets = jets[j_isclean]


        output['njet15'].fill(dataset=dataset, njet15=ak.num(good_jets))

        #print("number of good jets:",ak.num(good_jets))


        two_jets = (ak.num(good_jets) >= 2)

        #vpt_cut =  (vpt>=260) & (vpt<=390)
        #vmass_cut = (vmass>=60) & (vmass<=120)

        #full_selection = two_lep & two_jets & vpt_cut & vmass_cut
        full_selection = two_lep & two_jets
        #full_selection = two_lep

        selected_events = events[full_selection]
        output['cutflow'][dataset]["selected_events"] += len(selected_events)

        j_2l2j = good_jets[full_selection]
        dijet = j_2l2j[:, 0] + j_2l2j[:, 1]

        #print("number of good jets full selection:",ak.num(j_2l2j))
        #print("Dijets:", len(dijet), dijet)
        dijet_pt = dijet.pt
        dijet_m  = dijet.mass
        dijet_dr = j_2l2j[:, 0].delta_r(j_2l2j[:, 1])
        #print("Dijet mass:", len(dijet_m), dijet_m)

        weight = selected_events.genWeight
        #print("weights:", len(weight), weight)
        #weight = np.ones(len(selected_events))

        output['dilep_m'].fill(dataset=dataset, dilep_m=ak.flatten(vmass[full_selection]), weight=weight)
        output['dilep_pt'].fill(dataset=dataset,  PDFwei="Default", dilep_pt=ak.flatten(vpt[full_selection]), weight=weight)


        output['dijet_m'].fill(dataset=dataset, dijet_m=dijet_m, weight=weight)
        output['dijet_pt'].fill(dataset=dataset, dijet_pt=dijet_pt, weight=weight)
        output['dijet_dr'].fill(dataset=dataset, dijet_dr=dijet_dr, weight=weight)

        for p in range(0,33):
            PdfWei = 2*selected_events.LHEPdfWeight[:,p]
            #PdfWei = selected_events.LHEPdfWeight[:,p] # FOR DY 2J 50_150 there is no factor 2
            output['dilep_pt'].fill(dataset=dataset, PDFwei=str(p), dilep_pt=ak.flatten(vpt[full_selection]), weight=weight*PdfWei)

        return output

    def postprocess(self, accumulator):
        return accumulator

def _pdfunc(arr):
    # From https://gist.github.com/hqucms/f71a0223e04452538ee2c8af7cfdf0a1
    if len(arr) == 33:
        # PDF4LHC15_nnlo_30_pdfas
        delta = arr - arr[0]
        pdfunc = np.sqrt(np.sum(delta[1:31] ** 2))
        asunc = (arr[32] - arr[31]) / 2
        return np.sqrt(pdfunc**2 + asunc**2)
    elif len(arr) == 103:
        # NNPDF31_nnlo_hessian_pdfas
        delta = arr - arr[0]
        pdfunc = np.sqrt(np.sum(delta[1:101] ** 2))
        asunc = (arr[102] - arr[101]) / 2
        return np.sqrt(pdfunc**2 + asunc**2)
    elif len(arr) == 100:
        # NNPDF30_nlo_as_0118
        lo, hi = np.percentile(arr, [16, 84])
        return (hi - lo) / 2
    elif len(arr) == 101:
        # NNPDF30_lo_as_0118
        lo, hi = np.percentile(arr[1:], [16, 84])
        return (hi - lo) / 2
    elif len(arr) == 102:
        # NNPDF30_nlo_nf_5_pdfas
        lo, hi = np.percentile(arr[:100], [16, 84])
        pdfunc = (hi - lo) / 2
        asunc = (arr[101] - arr[100]) / 2
        return np.sqrt(pdfunc**2 + asunc**2)
    else:
        print("array:", arr, "length: ", len(arr))
        raise NotImplementedError

def printIntegrals(h, obs):
    ints = h.integrate(obs)
    print(ints, ints.values())
    yields = {}
    for key,v in ints.values().items():
        # print(key, key[0], key[1], v)
        sample = key[0]
        wei_id = key[1]
        if wei_id == "Default":
            yields[sample] = []
        else:
            yields[sample].append(v)
    print(yields)
    return yields

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
        if observable=="dilep_pt":
            hist.plotgrid(histogram, overlay='PDFwei', col='dataset', line_opts={})
            yi = printIntegrals(histogram, observable)
            for k,y in yi.items():
                #print(k,y)
                pdfunc = _pdfunc(np.array(y))
                print (k, "Uncertainty: %.1f %%"%(pdfunc/yi[k][0]*100))
        else:
            hist.plot1d(histogram, overlay='dataset', line_opts={}, overflow='none')
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
    parser = argparse.ArgumentParser(description='Run quick plots from NanoAOD input files')
    #parser.add_argument("inputfile")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument('--pkl', type=str, default=None,  help="Make plots from pickled file.")

    opt = parser.parse_args()

    print(opt)

    #from dask.distributed import Client
    import time

    #client = Client("tls://localhost:8786")
    #ntuples_location = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/NanoGEN/"
    #ntuples_location = "/net/data_cms/institut_3a/NanoGEN/"

    ntuples_location1 = "root://xrootd-cms.infn.it//store/mc/RunIIFall17NanoAODv7/"

    p2017_DY1_50_150 =  ntuples_location1 + "/DY1JetsToLL_M-50_LHEZpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/"
    p2017_DY2_50_150 = ntuples_location1 + "/DY2JetsToLL_M-50_LHEZpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/"

    p2017_DY1_150_250 = ntuples_location1 + "/DY1JetsToLL_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/230000/"
    p2017_DY2_150_250 = ntuples_location1 + "/DY2JetsToLL_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/100000/"

    p2017_DY1_150_250 = ntuples_location1 + "/DY1JetsToLL_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/230000/"
    p2017_DY2_150_250 = ntuples_location1 + "/DY2JetsToLL_M-50_LHEZpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_new_pmx_102X_mc2017_realistic_v8-v1/100000/"

    p2017_DY1_400_Inf = ntuples_location1 + "/DY1JetsToLL_M-50_LHEZpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/130000/"
    p2017_DY2_400_Inf = ntuples_location1 + "/DY2JetsToLL_M-50_LHEZpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/"

    ntuples_location2 = "root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/DYCOPY_NanoV7/"
    p2017_DY1_250_400 = ntuples_location2 + "/DY1JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/"
    p2017_DY2_250_400 = ntuples_location2+ "/DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/"

    file_list = {
        #'2017_DY1J_250_400' : getRootFiles(p2017_DY1_250_400),
        #'2017_DY2J_250_400' : getRootFiles(p2017_DY2_250_400),
        '2017_DY1J_50_150' : [p2017_DY1_50_150+"/22DD7F85-BE97-5E4D-9812-26F1471B7B8F.root"],
        #'2017_DY2J_50_150' : [p2017_DY2_50_150+"/473E794B-3180-3043-94D9-9CF5F4510C35.root"],
        '2017_DY1J_150_250' : [p2017_DY1_150_250+"/4222513F-35BA-2A42-AE6F-04B1BC368378.root"],
        '2017_DY2J_150_250' : [p2017_DY2_150_250+"/048B20FA-ADDE-8442-89E4-5399A019A7F9.root"],
        '2017_DY1J_250_400' : [p2017_DY1_250_400+"/B9101D62-5158-7649-8121-E3E3645EBA8A.root"],
        '2017_DY2J_250_400' : [p2017_DY2_250_400+"/18A3F63D-3CD9-2449-AC01-D14789684D8D.root"],
        '2017_DY1J_400_Inf' : [p2017_DY1_400_Inf + "/6AF6827B-7A67-0745-8CBF-D79CE3B33CEF.root"],
        '2017_DY2J_400_Inf' : [p2017_DY2_400_Inf + "/00F23B18-1177-7548-BB64-A5642F2D0CA9.root"],
    }
    # print(file_list)

    if opt.pkl!=None:
        plotFromPickles(opt.pkl, opt.outdir)
    else:
        output = processor.run_uproot_job(file_list,
                                          treename = 'Events',
                                          processor_instance = Processor(),
                                          #executor = processor.iterative_executor,
                                          #executor_args = {"schema": NanoGENSchema},
                                          #executor_args = {"schema": NanoAODSchema},
                                          #executor_args = {"schema": NanoAODPPSchema},
                                          executor = processor.futures_executor,
                                          executor_args = {'schema': NanoAODPPSchema, "workers":10}
                                      )



        plot(output, opt.outdir)


        for key, value in output['cutflow'].items():
            print(key, value)
            for key2, value2 in output['cutflow'][key].items():
                print(key, key2,value2)
        for key, value in output['sumw'].items():
            print(key, value)
