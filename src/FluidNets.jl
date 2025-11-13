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

include("hyperparameter_optimization.jl")
include("model.jl")
include("process_data.jl")

include("macros/get_files.jl")

export K_labels


end