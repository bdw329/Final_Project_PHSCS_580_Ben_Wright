# Final_Project_PHSCS_580_Ben_Wright
Predicting Lift to Drag Ratio of Reentry Capsule - Residual Model of Experimental Data and Modified Newtonian Theory

# PROBLEM STATEMENT:
The goal of this project is to develop an approximate model called the Modified Newtonian Hypersonic Method to approximate experimental data of Lift to Drag Ratio vs. α of the Orion reentry capsule at Mach 6, then use linear regression on a residual model to create a more accurate regression model of the experimental data. Experimental data is from the paper “Orion Crew Module Aerodynamic Testing” on nasa.gov, tested at Mach 6.

# CONTENTS OF FILES:
main.py: Contains the Modified Newtonian Law function, creates regression models and plots.
data.py: Contains experimental Lift_to_Drag data vs. angle of attack for Orion from NASA paper at Mach 6
capsule_geometry.py: Generates the outline of a 2D cross section of the Orion 
*All files are compatible with python 3.11.9*

# REQUIRED PACKAGES
numpy
matplotlib.pyplot
