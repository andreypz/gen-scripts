#!/usr/bin/env python3

from os import listdir, makedirs, path, system
#import numpy as np
import sys
import pickle as pkl
import subprocess
import uproot

def getFilesFromDas(ds):
    allfiles = str(subprocess.check_output(str('/cvmfs/cms.cern.ch/common/dasgoclient -query="file dataset=%s"'%ds.strip()), 
                                           shell=True), 'utf-8').split("\n")
    #print(ds, allfiles)
    return allfiles

def makeDasQueryAndPickle(inpfile):

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


    pklFileName = "./FilesOnDas.pkl"
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
    #print(info)

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
    parser = argparse.ArgumentParser(description='Run quick plots from NanoAOD input files')
    parser.add_argument("inputfile")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument('--pkl', type=str, default=None,  help="Make plots from pickled file.")

    opt = parser.parse_args()

    print(opt)
    

    if ".pkl" in opt.inputfile:
        full_file_list = pkl.load(open(opt.inputfile,'rb'))
    else:
        full_file_list = makeDasQueryAndPickle(opt.inputfile)

    print("All samples with query:", full_file_list.keys())

    sampleInfo = ReadSampleInfoFile('samples_2017_vhcc.txt')

    print("all processes:", sampleInfo.keys())
    
    xroot = 'root://xrootd-cms.infn.it/'

    file_list = makeListOfInputRootFilesForProcess("W1Jets-Pt50To150", sampleInfo, "./FilesOnDas.pkl", xroot, 2)
    print(file_list)


    file_list = makeListOfInputRootFilesForProcess("TT_DiLep", sampleInfo, "./FilesOnDas.pkl", xroot, 2)
    print(file_list)

    file_list = makeListOfInputRootFilesForProcess("WZTo3L1Nu", sampleInfo, "./FilesOnDas.pkl", xroot, 2)
    print(file_list)
