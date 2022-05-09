# Plotting scripts for LHE, NanoGEN and NanoAOD events

### LHE analyzer and plotter
Here are a few scripts to anlyze LHE events with python and NanoGEN events with coffea. It is originally copied from https://github.com/AndreasAlbert/mg5tut_apr21_plots
* To set it up run this:

```
source env.sh
pip install lhereader matplotlib mplhep 
pip install --upgrade dask
pip install --upgrade numba
pip install coffea 

```

* Then after each setup, you have to run `source env.sh` again.

* Run the LHE analyzer:
```
./lhePlot.py /PathToLHE/cmsgrid_final.lhe -o outDir
```

###  NanoGEN analyser
This is a script to analyze NanoGEN events with coffea, as well as simple loop.
* Run NanoGen analyzer:
```
./nanoGenAna.py /PathTo/NanoGEN/Tree_1.root [-l -o outDir]

```

###  NanoAOD analyser for V+Jets study
For this you may need to set up a new environment with conda, installing missing packages (`parsl` and maybe more). This part is not covered in these instructions.
* After setting the conda environment run this:
```
./cofeGeno.py -n 1 [-o plots_output_dir] [-e parsl] [-t pre]
```
it will run over one file (`-n 1` option) of each dataset in the file list (hardcoded).

* In order to create a list of files for datasets run this:
```
./sampleInfo.py mc_vjets_samples.list -o SamplePickleFile.pkl
```