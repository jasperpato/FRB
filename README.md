# Can Fast Radio Bursts Probe the Ionized Gas in Distant Galaxies?

## ICRAR Summer Project 2023/24

Jasper Paterson  
Apurba Bera, Marcin Glowacki, Clancy James

The scripts in this package:

- curve fit high time resolution FRB data
- plot curve fits
- generate FRB property data and collate host galaxy data
- perform correlations between FRB properties and host galaxy properties
- plot correlations

How to:

1. populate the data/pkls file with high time resolution (HTR) FRB pickle data
2. create a virtual environment

   - `python3 -m venv venv`
   - `source venv/bin/activate`
   - `pip3 install -r req.txt`

3. for each submodule
   - change into submodule directory, eg. `cd astropath`
   - `python3 setup.py install`
4. `python3 curve_fit.py`
5. `python3 plot_fit.py`
6. `python3 populate_table.py`
7. `python3 correlate.py`

After this process, fitted curve plots can be found in figs/fits, and correlation scatter plots can be found in figs/correlations.

Submodules

- https://github.com/FRBs/astropath
- https://github.com/FRBs/FRB
- https://github.com/FRBs/ne2001

Versions:

- Python 3.11.4
- numpy 1.26.2
- scipy 1.11.3

TODO:

- finish automatic host galaxy data scraping from FRB package (populate_table.py)
