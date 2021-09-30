#!/usr/bin/env python3
import uproot
#uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource
#xrootd_handler=uproot.MultithreadedXRootDSource

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
import sampleInfo as si

def getRootFiles(d, xroot, dasgo=True, lim=None):
    import subprocess
    if "xrootd" in d:
        sp = d.split("/")
        siteIP = "/".join(sp[0:3])
        pathToFiles = "/".join(sp[3:-1])
        allfiles = str(subprocess.check_output(["xrdfs", siteIP, "ls", pathToFiles]), 'utf-8').split("\n")
        rootfiles = [siteIP+f for i,f in enumerate(allfiles) if f.endswith(".root") and (lim==None or i<lim)]
    elif dasgo:
        allfiles = str(subprocess.check_output(str('/cvmfs/cms.cern.ch/common/dasgoclient -query="file dataset=%s"'%d), shell=True), 'utf-8').split("\n")
        rootfiles = [] 
        for f in allfiles:
            try:
                with uproot.open(xroot+f) as up:
                    #print("Uproot can open file, ", f)
                    rootfiles.append(xroot+f)
            except:
                #print("No, we can't open file, ", f)
                pass
            if lim!=None and len(rootfiles)==lim:
                #print("It's enough files")
                break
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
                 "channel": hist.Cat("channel", "channel"),
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
            {observable : hist.Hist("Counts", axis["dataset"], var_axis) for observable, var_axis in axis.items() if observable not in ["dataset", "channel", "PDFwei", "dijet_pt"]}
        )
        self._accumulator['dijet_pt'] = hist.Hist("Counts", axis["dataset"], axis["channel"], axis["PDFwei"], axis["dijet_pt"])
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
        print("dataset:", dataset, "PDF LHE weights:", len(events.LHEPdfWeight), np.mean(events.LHEPdfWeight), events.LHEPdfWeight)
        print(np.mean(ak.num(events.LHEPdfWeight)), ak.num(events.LHEPdfWeight))

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
        #nmuons = ak.sum(goodmuon, axis=1)

        #lead_muon_pt = ak.firsts(muons[goodmuon]).pt > 20

        muons = muons[goodmuon]

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

        muons = ak.with_field(muons, 0, 'flavor')
        electrons = ak.with_field(electrons, 1, 'flavor')
        
        leptons = ak.with_name(ak.concatenate([muons, electrons], axis=1), 'PtEtaPhiMCandidate')
        
        nlep = ak.num(leptons)

        dileptons = ak.combinations(leptons, 2, fields=['i0', 'i1'])


        # mu = 0, e = 1, so: mumu = 0, emu = 1, ee = 2.
        #di_type = (leptons[dileptons['i0']].flavor + leptons[dileptons['i1']].flavor)
        OS = (dileptons['i0'].charge != dileptons['i1'].charge)
        SF = (dileptons['i0'].flavor == dileptons['i1'].flavor)
        #dileptonMask = (ak.num(dileptons) == 1) & dileptons['i0'].flavor==dileptons['i1'].flavor & dileptons['i0'].charge != dileptons['i1'].charge & ak.any((leptons[dileptons['i0']].pt > 25) | (leptons[dileptons['i1']].pt > 25), axis=1) & ( ((dileptons['i0'] + dileptons['i1']).mass - 91.19) < 15) 
        pt25  = ((dileptons['i0'].pt > 25) | (dileptons['i1'].pt > 25))
        Zmass = (((dileptons['i0'] + dileptons['i1']).mass - 91.19) < 15)
        dileptonMask = OS & SF & pt25 & Zmass
        good_dileptons = dileptons[dileptonMask]
        
        #ch_2mu = tight_ll[ak.sum(tight_ll.flavor, axis=1) == 0]
        #ch_2e = tight_ll[ak.sum(tight_ll.flavor, axis=1) == 2

        vpt = (good_dileptons['i0'] + good_dileptons['i1']).pt
        vmass = (good_dileptons['i0'] + good_dileptons['i1']).mass

        output['nlep'].fill(dataset=dataset, nlep=nlep)

        two_lep = ak.num(good_dileptons) == 1
        one_lep = ak.num(leptons) == 1
        zero_lep = ak.num(leptons) == 0

        #print(good_dileptons[two_lep])
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
        two_jets = (ak.num(good_jets) >= 2)

        #j_2l2j = good_jets[full_selection_2L]
        #dijet = j_2l2j[:, 0] + j_2l2j[:, 1]

        output['njet15'].fill(dataset=dataset, njet15=ak.num(good_jets))

        #print("number of good jets:",ak.num(good_jets))


        #vpt_cut =  (vpt>=260) & (vpt<=390)
        #vmass_cut = (vmass>=60) & (vmass<=120)

        #full_selection_2L = two_lep & two_jets & vpt_cut & vmass_cut
        full_selection_2L = two_lep & two_jets
        full_selection_1L = one_lep & two_jets
        full_selection_0L = zero_lep & two_jets
        #full_selection_2L = two_lep

        for ch in ["2L","1L","0L"]:
            if ch=="2L": selection = full_selection_2L
            if ch=="1L": selection = full_selection_1L
            if ch=="0L": selection = full_selection_0L
            selected_events = events[selection]
            output['cutflow'][dataset]["selected_events_"+ch] += len(selected_events)

            dijets = jets[selection]

            dijet = dijets[:, 0] + dijets[:, 1]

            #print("number of good jets full selection:",ak.num(j_2l2j))
            #print("Dijets:", len(dijet), dijet)
            dijet_pt = dijet.pt
            dijet_m  = dijet.mass
            dijet_dr = dijets[:, 0].delta_r(dijets[:, 1])
            #print("Dijet mass:", len(dijet_m), dijet_m)

            weight = selected_events.genWeight
            #print("weights:", len(weight), weight)
            #weight = np.ones(len(selected_events))

            if ch=="2L":
                output['dilep_m'].fill(dataset=dataset, dilep_m=ak.flatten(vmass[selection]), weight=weight)
                output['dilep_pt'].fill(dataset=dataset, dilep_pt=ak.flatten(vpt[selection]), weight=weight)

            output['dijet_m'].fill(dataset=dataset, dijet_m=dijet_m, weight=weight)
            output['dijet_dr'].fill(dataset=dataset, dijet_dr=dijet_dr, weight=weight)

            output['dijet_pt'].fill(dataset=dataset, channel=ch, PDFwei="Default", dijet_pt=dijet_pt, weight=weight)
            nPDFs = int(np.mean(ak.num(events.LHEPdfWeight)))
            meanPDF = np.mean(events.LHEPdfWeight)
            for p in range(0,nPDFs):
                if abs(0.5-meanPDF)<0.15:
                    # PDF weights are off by a factor 2 
                    PdfWei = 2*selected_events.LHEPdfWeight[:,p]
                else:
                    PdfWei = selected_events.LHEPdfWeight[:,p]
                output['dijet_pt'].fill(dataset=dataset, channel=ch, PDFwei=str(p), dijet_pt=dijet_pt, weight=weight*PdfWei)

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
        print(key, key[0], key[1], v)
        sample = key[0]
        chan  = key[1]
        wei_id = key[2]
        if wei_id == "Default":
            yields[chan] = {}
            yields[chan][sample] = []
        else:
            yields[chan][sample].append(v)
    print(yields)
    return yields

def plot(histograms, outdir, fromPickles=False):
    '''Plots all histograms. No need to change.'''
    if not path.exists(outdir):
        makedirs(outdir)

    if not fromPickles:
        pkl.dump( histograms,  open(outdir+'/Pickles.pkl',  'wb')  )

    for observable, histogram in histograms.items():
        #print (observable, histogram, type(histogram))
        if type(histogram) is hist.hist_tools.Hist:
            print(observable, "I am a Hist", histogram)
        else:
            continue
        plt.gcf().clf()
        if observable=="dijet_pt":
            hist.plotgrid(histogram, overlay='PDFwei', col='dataset', row='channel', line_opts={})
            yi = printIntegrals(histogram, observable)
            for ch,y1 in yi.items():
                for s,y2 in y1.items():
                    print("Channael:", ch, "sample=", y2)
                    pdfunc = _pdfunc(np.array(y2))
                    print (ch, s, "Uncertainty: %.1f %%"%(pdfunc/yi[ch][s][0]*100))
        else:
            hist.plot1d(histogram, overlay='dataset', line_opts={}, overflow='none')
        plt.gca().autoscale()
        plt.gcf().savefig(f"{outdir}/{observable}.png")


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
    parser.add_argument('-n','--numberOfFiles', type=int, default=1,  help="Number of files to process per sample")

    opt = parser.parse_args()

    print(opt)

    if opt.pkl!=None:
        plotFromPickles(opt.pkl, opt.outdir)
    else:

        import time
        
        #xroot = 'root://xrootd-cms.infn.it/'
        xroot = 'root://cms-xrd-global.cern.ch/'
        
        sampleInfo = si.ReadSampleInfoFile('2L_samples_2017_vhcc.txt')
        
        file_list_DY = {ds: si.makeListOfInputRootFilesForProcess(ds, sampleInfo, "./FilesOnDas.pkl", xroot, lim=opt.numberOfFiles) for ds in [
            #'DY1ToLL_PtZ-50To150',
            #'DY2ToLL_PtZ-50To150',
            #'DY1ToLL_PtZ-150To250',
            #'DY2ToLL_PtZ-150To250',
            #'DY1ToLL_PtZ-250To400',
            #'DY2ToLL_PtZ-250To400',
            #'DY1ToLL_PtZ-400ToInf',
            #'DY2ToLL_PtZ-400ToInf',
        ]
                    }
        
        file_list_other = {ds: si.makeListOfInputRootFilesForProcess(ds, sampleInfo, "./FilesOnDas.pkl", xroot, lim=opt.numberOfFiles) for ds in [
            #'ZH125ToCC_ZLL_powheg',
            #'TT_DiLep',
            #'TT_SingleLep',
            #'TT_AllHadronic'
        ]
                       }
        
        file_list_all = {ds: si.makeListOfInputRootFilesForProcess(ds, sampleInfo, "./FilesOnDas.pkl", xroot, lim=opt.numberOfFiles, checkOpen=True) for ds in sampleInfo.keys()}
        
        #file_list = file_list_DY
        #file_list = file_list_other
        file_list = file_list_all
        
        #file_list = { "p2017_DY2_250_400": ["root://grid-cms-xrootd.physik.rwth-aachen.de//store/user/andrey/DYCOPY_NanoV7/DY2JetsToLL_M-50_LHEZpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8/NANOAODSIM/PU2017_12Apr2018_Nano02Apr2020_102X_mc2017_realistic_v8-v1/100000/470F9AB8-2D2B-AC47-832C-14D4EBF9DAD6.root"]}
        print(file_list)
         
        output = processor.run_uproot_job(file_list,
                                          treename = 'Events',
                                          processor_instance = Processor(),
                                          #executor = processor.iterative_executor,
                                          #executor_args = {"schema": NanoGENSchema},
                                          #executor_args = {"schema": NanoAODSchema},
                                          #executor_args = {"schema": NanoAODPPSchema},
                                          executor = processor.futures_executor,
                                          executor_args = {'schema': NanoAODSchema, "workers":6},# "xrootdtimeout": 10},# "skipbadfiles": True},
                                          #maxchunks=opt.numberOfFiles
                                      )



        plot(output, opt.outdir)


        for key, value in output['cutflow'].items():
            print(key, value)
            for key2, value2 in output['cutflow'][key].items():
                print(key, key2,value2)
        for key, value in output['sumw'].items():
            print(key, value)
