# FluidNets.jl
Structures to build, train and optimize Neural Networks to recreate Kernelfunctions as used in Fluidum.  


The file 'tutorial.jl' contains a workflow to introduce the functions used to train a simple neural network and show the output.  


The content is structured as follows:  

1) general_functions  
contain the general functions of FluidNets that can be used as they are. Their application is exemplary shown in the example_scripts.  


2) customized_scripts  
scripts for special purposes that define functions which might have to be customized.  
'get_files' contains a structure to convert FastReso files into files as used in the rest of the FluidNets-code.  
'run_trials' contains an example on how to run the hyperparameter optimization.  
'minifluidum' contains a minimal use of Fluidum to calculate a freeze-out, which can then be used to compare spectra from network predictions.  


3) example_scripts  
tutorial-like scripts that use the general function. Can be used as introduction or general workflow.  
'make_model' contains a structure on how to build a neural network.  
'evaluate_trials' - work in progress
