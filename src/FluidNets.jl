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
using Fluidum

include("general_functions/model.jl")
include("general_functions/process_data.jl")
include("general_functions/hyperparameter_optimization.jl")
include("general_functions/plotstuff.jl")


# custom scripts - not necessary
include("customized_scripts/get_files.jl")
include("customized_scripts/run_trials.jl")
include("customized_scripts/minifluidum.jl")



export model_structure, model_structures, initiate_model, train_model!, reprocess_model, save_model, load_model
export read_data, get_mean_std, get_mid_halfwidth, get_min_width, get_zero_absmax, preprocess, get_train_test_set
export extrapolate_interpolate_kernels, Kernels
export MyXweightLoss, MyYweightLoss, leakyrelu_grad
export trials, run_all_trials, save_trials, load_trials, merge_trials
export plot_losses, plot_sorted_kernels_ptur, plot_sorted_kernels_temps, compare_kernels_ptur, compare_kernels_temps



end