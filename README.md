# Plotting scripts for LHE and NanoGEN

Here are a few scripts to anlyze LHE events with python and NanoGEN events with coffea.
To set it up run this:

```bash 
source env.sh

pip install lhereader matplotlib mplhep 
pip install --upgrade dask
pip install --upgrade numba
pip install coffea 

```

Then after each setup, you have to run `source env.sh` again.


Run the LHE analyzer:
```
./lhePlot.py /PathToLHE/cmsgrid_final.lhe -o outDir
```

Run NanoGen analyzer:
```
./nanoGenAna.py /PathTo/NanoGEN/Tree_1.root [-l -o outDir]

```