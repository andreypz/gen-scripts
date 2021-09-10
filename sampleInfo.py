#!/usr/bin/env python3

#from os import listdir, makedirs, path, system
#import numpy as np
import pickle as pkl
import subprocess

def getFilesFromDas(ds):
    allfiles = str(subprocess.check_output(str('/cvmfs/cms.cern.ch/common/dasgoclient -query="file dataset=%s"'%ds.strip()), 
                                           shell=True), 'utf-8').split("\n")
    #print(ds, allfiles)
    return allfiles

def MakePickle(inpfile):

    allFiles = {}
    with open(inpfile, mode='r') as in_file:

        for line in in_file:
            if line.startswith('#') or line.strip()=="":
                #print("we skip this line:", line)
                continue
            sample = line
            dsname = sample.split("/")[1]
            if dsname in allFiles.keys():
                allFiles[dsname].extend(getFilesFromDas(sample))
            else:
                allFiles[dsname] = getFilesFromDas(sample)
    #print(allFiles)


    pklFileName = "./FilesOnDas.pkl"
    pkl.dump( allFiles,  open(pklFileName,  'wb')  )
    print("The files are written to ", pklFileName)

def ReadSampleInfoFile(inpfile):
     with open(inpfile, mode='r') as in_file:

        for line in in_file:
            if not line.startswith('name'):
                continue
            else:
                print(line)

if __name__ == "__main__":
    print("This is the __main__ part")

    import argparse
    parser = argparse.ArgumentParser(description='Run quick plots from NanoAOD input files')
    parser.add_argument("inputfile")
    parser.add_argument('-o','--outdir', type=str, default="plots_default", help="Directory to output the plots.")
    parser.add_argument('--pkl', type=str, default=None,  help="Make plots from pickled file.")

    opt = parser.parse_args()

    print(opt)

    MakePickle(opt.inputfile)

    file_list = pkl.load(open("./FilesOnDas.pkl",'rb'))

    print(file_list)

    
    ReadSampleInfoFile('samples_2017_vhcc.txt')
