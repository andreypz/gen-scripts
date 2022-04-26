#!/usr/bin/env python3

from os import listdir, makedirs, path, system
#import numpy as np
import sys
import pickle as pkl
import subprocess
import uproot

xs_150_250 = {
    "2016_DYnJ": 88,
    "2017_DY1J": 9.7,
    "2017_DY2J": 16.0,
}
xs_250_400 = {
    #"2016_DYnJ": 3.13,  # TuneCUET8M1
    "2016_DYnJ": 3.67,  # TuneCP5
    "2017_DY1J": 1.13,
    "2017_DY2J": 2.71,
}

def getRootFilesFromPath(d, lim=None):
    import subprocess
    if "xrootd" in d:
        sp = d.split("/")
        siteIP = "/".join(sp[0:4])
        pathToFiles = "/".join(sp[3:-1])+"/"
        allfiles = str(subprocess.check_output(["xrdfs", siteIP, "ls", pathToFiles]), 'utf-8').split("\n")
        #rootfiles = [siteIP+'/'+f for i,f in enumerate(allfiles) if f.endswith(".root") and (lim==None or i<lim)]
    else:
        siteIP = ""
        pathToFiles = d
        allfiles = [path.join(d, f) for i,f in enumerate(listdir(d)) if f.endswith(".root")]

    rootfiles = []
    for file_or_dir in allfiles:
        # print(file_or_dir)
        if (file_or_dir == "" or file_or_dir == pathToFiles): continue
        file_or_dir = siteIP + file_or_dir
        if file_or_dir.endswith(".root"):
            if lim==None or len(rootfiles)<lim:
                rootfiles.append(file_or_dir)
                
        elif not "log" in file_or_dir and not file_or_dir[-1]=='/' and  len(rootfiles)<lim:
            file_or_dir=file_or_dir+'/'
            rootfiles.extend(getRootFilesFromPath(file_or_dir, lim-len(rootfiles)))

    print("Input path:", d)
    print("List of root files to be processed:\n",rootfiles)

    return rootfiles

def getFilesFromDas(ds):
    allfiles = str(subprocess.check_output(str('/cvmfs/cms.cern.ch/common/dasgoclient -query="file dataset=%s"'%ds.strip()), 
                                           shell=True), 'utf-8').split("\n")
    #print(ds, allfiles)
    return allfiles

def makeDasQueryAndPickle(inpfile, pklFileName="./FilesOnDas.pkl"):

    allFiles = {}
    with open(inpfile, mode='r') as in_file:

        for line in in_file:
            if line.startswith('#') or line.strip()=="":
                #print("we skip this line:", line)
                continue
            sample = line.strip()
            dsname = sample.split("/")[1]
            print("Getting file list for sample=%s, ds=%s"%(sample,dsname))
            if dsname in allFiles.keys():
                allFiles[dsname].extend(getFilesFromDas(sample))
            else:
                allFiles[dsname] = getFilesFromDas(sample)
            print("Found %i files"%len(allFiles[dsname]))
    # 
    # print(allFiles)

    pkl.dump( allFiles,  open(pklFileName,  'wb')  )
    print("The files are written to ", pklFileName)
    return allFiles

def ReadSampleInfoFile(inpfile):
    
    info = {}
    with open(inpfile, mode='r') as in_file:
        for line in in_file:
            sampleInfo = {}
            name = None
            if not line.startswith('name'): continue
            for item in line.split():
                # print("Item: ", item)
                if "=" not in item: continue
                key,value=item.split("=")
                if key=="name":
                    name=str(value)
                if key=="dir":
                    sampleInfo['dataset'] = str(value).strip("/")
                if key=="type":
                    sampleInfo["type"]=int(value)
                if key=="xsec":
                    sampleInfo["xsec"]=float(value)
                if key=="kfac":
                    sampleInfo["kfac"]=float(value)

            info[name] = sampleInfo
    # 
    # print(info)

    return info

def makeListOfInputRootFilesForProcess(proc_name, sampleInfo, listOfFilesPkl, xroot, lim=None, checkOpen=False):

    if proc_name not in sampleInfo.keys():
        print("This process does not exist on the books:", proc_name)
        sys.exit(1)
    
    dsname = sampleInfo[proc_name]["dataset"]
    print("%s process has the dataset name %s"%(proc_name, dsname) )

    file_list = pkl.load(open(listOfFilesPkl,'rb'))

    if dsname not in file_list.keys():
        print("This dataset name does not exist on the books:", dsname)
        return []

    allfiles = file_list[dsname]
    # print("all files for sample %s: \n"%dsname, allfiles)
    rootfiles = []
    for f in allfiles:
        if f.strip()=='': continue
        fname = xroot+f
        if checkOpen:
            try:
                with uproot.open(fname) as up:
                    #print("Uproot can open file, ", fname)
                    rootfiles.append(fname)
            except:
                print("No, we can't open the file: ", fname)
                pass
        else:
            rootfiles.append(fname)

        if lim!=None and len(rootfiles)==lim:
            #print("It's enough files")
            break

    #print(rootfiles)
    return rootfiles


if __name__ == "__main__":
    print("This is the __main__ part")

    import argparse
    parser = argparse.ArgumentParser(description='Run das quiries etc')
    parser.add_argument("inputfile")
    parser.add_argument('-o','--output', type=str, default="./FilesOnDas.pkl", help="Directory to output the plots.")

    opt = parser.parse_args()

    print(opt)
    

    if ".pkl" in opt.inputfile:
        full_file_list = pkl.load(open(opt.inputfile,'rb'))
    else:
        full_file_list = makeDasQueryAndPickle(opt.inputfile, opt.output)

    print("All samples with query:", full_file_list.keys())

    #sampleInfo = ReadSampleInfoFile('2L_samples_2017_vhcc.txt')
    #sampleInfo = ReadSampleInfoFile('mc_2016_vhcc.conf')
    sampleInfo = ReadSampleInfoFile('mc_vjets_samples.info')

    print("all processes:", sampleInfo.keys())
    
    #xroot = 'root://xrootd-cms.infn.it/'
    #xroot = 'root://cms-xrd-global.cern.ch/'
    xroot = 'root://grid-cms-xrootd.physik.rwth-aachen.de/'
   
    pkl_file = "./VJetsPickle.pkl"
    #file_list = makeListOfInputRootFilesForProcess("DYJets_inc_FXFX", sampleInfo, pkl_file, xroot, lim=2, checkOpen=True)
    #file_list = makeListOfInputRootFilesForProcess("DYJets_inc_MLM", sampleInfo, pkl_file, xroot, lim=2, checkOpen=True)
    file_list = makeListOfInputRootFilesForProcess("DYJets_inc_MinNLO", sampleInfo, pkl_file, xroot, lim=2, checkOpen=True)
    print(file_list)

    """
    pkl = "./FilesOnDas_2016.pkl"
    file_list = makeListOfInputRootFilesForProcess("W1Jets-Pt50To150", sampleInfo, pkl, xroot, 2)
    print(file_list)


    file_list = makeListOfInputRootFilesForProcess("TT_DiLep", sampleInfo, pkl, xroot, 2)
    print(file_list)

    file_list = makeListOfInputRootFilesForProcess("WZTo3L1Nu", sampleInfo, pkl, xroot, 2)
    print(file_list)
    """
