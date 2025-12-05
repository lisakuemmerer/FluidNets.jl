using FluidNets

##############################################################################################################
# This script runs hyperparameter optimization.
# the 'trial' function will be called for different options defined below by the 'trials' function

# path to where your dataset
in_file = "/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt"
# path to where you want to save output
out_file = "/home/lisa/MA/NeuralNetwork/hyperparameter/testtrial.jld2"
#number of variables & Kernels
var_dim, K_dim = 4,8


################################################################################################################

# read in data
var_set, K_set = read_data(in_file, var_dim, K_dim);


# put in hyperparameter options you wish to try
# function can not be saved correctly, which is why strings are used @ frontend. options are explained in scenario_frontend_to_backend and can be adjusted there
# if you want to evaluate different hyperparameters you need to adjust the trial function accordingly
scens = Dict{Symbol, Any}(:prep_vars => ["none", "minwidth", "midhalfwidth", "meanstd"],
    :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax", "meanstd"], 
    :nb_hl => [4,5,6], 
    :hl_dim => [32,64,128], 
    :act_fct => ["sigmoid","tanh", "relu", "leakyrelu_001", "leakyrelu_01"], 
    :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "nothing"],
    :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing", "zeros"],
    :batchsize => [100,500,1000], 
    :lera => collect(0.0005:0.0005:0.005),
    :beta1 => [0.8,0.85,0.9,0.95,0.99], 
    :beta2 => [0.99,0.999,0.9999], 
    :lambda => [0.0,0.01,0.02]);



# put in exeptions for combinations of parameter choices that do not make sense
excepts = [s->s[:loss_fct]=="yweight" && (s[:prep_K]=="midhalfwidth" || s[:prep_K]=="minwidth" || s[:prep_K]=="meanstd"), 
    s -> s[:loss_fct]=="xweight" && (s[:prep_vars]=="none" || s[:prep_vars]=="midhalfwidth" || s[:prep_vars]=="meanstd")];


# scens_fast = Dict{Symbol, Any}(:prep_vars => ["none", "minwidth", "midhalfwidth"],
#     :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax"], 
#     :nb_hl => [4], 
#     :hl_dim => [32], 
#     :act_fct => ["sigmoid", "relu", "leakyrelu_01", "leakyrelu_02"], 
#     :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
#     :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
#     :batchsize => [100], 
#     :loss_fct => ["mse", "xweight", "yweight"], 
#     :lera => 10 .^LinRange(-4,-2,10),
#     :beta1 => [0.8,0.85,0.9,0.95,0.99], 
#     :beta2 => [0.99,0.999,0.9999], 
#     :lambda => [0.0,0.01,0.02])

# change lera !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



# # DO NOT UNCOMMENT UNLESS WANTED, RUNS FOR MULTIPLE HOURS !!!
# # function will call trial-function and run #num_trials options from scen without excepts
trial_all = trials(scens, var_set, K_set, excepts=excepts, num_trials=2);


# save the trials
save_trials(trial_all, out_file);



