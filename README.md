<p align="center">
  <img width="100" height="100" src=".logo.PNG">
</p>

# **Notes on running scripts**
## Staging DNS Data - dataSetup
   - PHASTA output files should be in `varts` dir
   - Run the `clean.py` script to eliminate duplicate probes. Creates files in `newvarts` dir
   - Run the `main.py` script to extract DNS data
     - Note that both `clean.py` and `main.py` scripts use `inputs.txt` file.
