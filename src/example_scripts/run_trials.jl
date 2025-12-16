using FluidNets
using Random
using LinearAlgebra
BLAS.set_num_threads(1)

##############################################################################################################
# This script runs hyperparameter optimization.
# the 'trial' function will be called for different options defined below by the 'trials' function

# path to where your dataset
in_file = "/home/lisa/MA/Data/Full_PCE/Kernels/pion_total_BG.txt";
# path to where you want to save output
out_path = "/home/lisa/MA/Final/hyperopt/pion_total/";
#number of variables & Kernels
var_dim, K_dim = 4, 8;


################################################################################################################

# read in data
var_set, K_set = read_data(in_file, var_dim, K_dim);

# put in hyperparameter options you wish to try
# function can not be saved correctly, which is why strings are used @ frontend. options are explained in scenario_frontend_to_backend and can be adjusted there
# if you want to evaluate different hyperparameters you need to adjust the trial function accordingly
scens = Dict{Symbol, Any}(:prep_vars => ["none", "minwidth", "midhalfwidth", "meanstd"],
    :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax"], 
    :nb_hl => [4,5,6,7], 
    :hl_dim => [32,64,128,256], 
    :act_fct => ["sigmoid", "tanh", "relu", "leakyrelu_001", "leakyrelu_01"], 
    :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "nothing"],
    :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "nothing", "zeros"],
    :batchsize => [100,500,1000], 
    :lera => collect(0.0001:0.0001:0.002),
    :beta1 => [0.8,0.85,0.9,0.95,0.99], 
    :beta2 => collect(range(0.99,0.9999,10)), 
    :lambda => collect(0.:0.005:0.02));



# put in exeptions for combinations of parameter choices that do not make sense
# excepts = [s->s[:loss_fct]=="yweight" && (s[:prep_K]=="midhalfwidth" || s[:prep_K]=="minwidth" || s[:prep_K]=="meanstd"), 
#     s -> s[:loss_fct]=="xweight" && (s[:prep_vars]=="none" || s[:prep_vars]=="midhalfwidth" || s[:prep_vars]=="meanstd")];



# # DO NOT UNCOMMENT UNLESS WANTED, RUNS FOR MULTIPLE HOURS !!!
# # function will call trial-function and run #num_trials options from scen without excepts
trial_all = trials(scens, var_set, K_set, num_trials=1280);


# save the trials to file with random name so that script can be rerun
isdir(out_path) || mkpath(out_path)
save_trials(trial_all, String(out_path*"trial_"*randstring("0123456789", 8)*".jld2"))
