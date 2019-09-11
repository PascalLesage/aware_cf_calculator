# aware_cf_calculator

Package to calculate AWARE (relative Available WAter REmaining per 
area in a watershed) characterization factors (CFs).

For more information on the AWARE method, visit theu WULCA website at 
http://www.wulca-waterlca.org/aware.html

The aware_cf_calculator package is written in Python and has three classes:
- AwareStatic, to calculate static (or deterministic) CFs.
- AwareStochastic, to calculate sets of CFs that account for the uncertainty of input parameters.
- AwareAnalysis, collecting an ad hoc suite of analysis tools 

A formatted Excel spreadsheet with data on input parameters is required and is found in the data directory.
