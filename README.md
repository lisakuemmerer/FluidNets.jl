# FluidNets.jl
Structures to build, train and optimize Neural Networks to recreate Kernelfunctions as used in Fluidum.



The content is structured as follows:

1) general_functions  
contains the general functions of FluidNets and can be used as is. Their application is exemplary shown in the example scripts.

2) customized_scripts  
example scripts that define special functions that might have to be customized.  
'get_files' contains a structure to convert FastReso files into files as used in the rest of the FluidNets-code.  
'run_trials' contains an example on how to run the hyperparameter optimization.

3) example_scripts  
tutorial-like scripts that use the general function. Can be used as introduction or general workflow.  
'make_model' contains a structure on how to build a neural network.  
'evaluate_trials' - work in progress
