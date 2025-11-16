module FluidNets

using DelimitedFiles
using Interpolations
using Random
using Lux
using Optimisers
using Zygote
using Plots
using JLD2
using Statistics
using Functors
using DataStructures
using StatsBase

include("general_functions/model.jl")
include("general_functions/process_data.jl")
include("general_functions/hyperparameter_optimization.jl")
#include("general_functions/plotstuff.jl")


# custom scripts - not necessary
include("customized_scripts/get_files.jl")
include("customized_scripts/run_trials.jl")

#include("example_scripts/make_model.jl")
#include("example_scripts/evaluate_trials.jl")



export model_structure, model_structures, initiate_model, train_model!, reprocess_model, save_model, load_model
export read_data, get_mean_std, get_mid_halfwidth, get_min_width, get_zero_absmax, preprocess, get_train_test_set
export extrapolate_interpolate_kernels, Kernels
export MyXweightLoss, MyYweightLoss, leakyrelu_grad
export trials, run_all_trials, save_trials, load_trials, merge_trials
export plot_losses


#custom scripts - not necessary
export particle_ids, K_labels, get_BG_mode_files
export trial, scen, excepts



end