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
using NearestNeighbors
using MLUtils

include("general_functions/build_model.jl")
include("general_functions/process_data.jl")
include("general_functions/hyperparameter_optimization.jl")
include("general_functions/plotstuff.jl")


# custom scripts - not necessary
include("customized_scripts/get_files.jl")
include("customized_scripts/minifluidum.jl")



export model_structure, model_structures, initiate_model, train_model!, reprocess_model, save_model, load_model, save_losses, load_losses
export read_data, get_mean_std, get_mid_halfwidth, get_min_width, get_zero_absmax, preprocess, get_train_test_set
export extrapolate_interpolate_kernels, Kernels
export MyXweightLoss, MyYweightLoss, leakyrelu_grad
export scenario_frontend_to_backend, trials, save_trials, load_trials, merge_trials, save_hyppars
export sortby, get_options, hist_loss, plot_hyppar, hist_hyppar, plot_course_in_best_trials, hist_occurance, plot_all_hyppars, plot_correlation,  hist_correlation_occurance, loop_one_2D, loop_all_2D
export plot_losses, plot_sorted_kernels_ptur, plot_sorted_kernels_temps, compare_kernels_ptur, compare_kernels_temps


end