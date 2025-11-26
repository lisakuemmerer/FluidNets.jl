

using FluidNets
using Fluidum
using Lux
using Plots


# this block contains all the parameters that need to be adjusted
var_dim, K_dim = 4,8; # number of variables & Kernels
datapath = "/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt"; # path to datafile
savepath = "/home/lisa/MA/NeuralNetwork/pion_4D_BG/tests/"; # path where output should be saved
saveas = "default_uniform"; # name indicating what kind of model this is
# choose hyperparatemers, options explained in workflow.jl
hyppars = Dict{Symbol, Any}(
    :n_train => 10000,
    :n_test => 10000,
    :uniform => true,
    :prep_vars => "minwidth",
    :prep_K => "meanstd", 
    :nb_hl => 5 ,
    :hl_dim => 128 ,
    :act_fct => "leakyrelu_01",
    :initializer_weight => "nothing",
    :initializer_bias => "zeros",
    :batchsize => 500,
    :nepochs => 1000,
    :early_stopping => false,
    :loss_fct => "yweight_01", 
    :lera => 0.001,
    :adapt_lera => false,
    :lera_trend => 0.999,
    :beta1 => 0.9, 
    :beta2 => 0.999, 
    :lambda => 0.0); 


    


# load data
var_set, K_set = read_data(datapath, var_dim, K_dim);

# define interpolation of kernels to be able to compare prediction
K_func = extrapolate_interpolate_kernels(var_set, K_set);

# translate chosen hyperparameters
scen = scenario_frontend_to_backend(hyppars);

# separate data in train & test sets, save preprocessing parameters
var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set,
preprocess_vars=scen[:prep_vars], preprocess_K=scen[:prep_K], n_train=scen[:n_train], n_test=scen[:n_test], uniform=scen[:uniform]);

# initialize model
my_NN = initiate_model(var_dim, K_dim, nb_hl=scen[:nb_hl], hl_dim=scen[:hl_dim], act_fct=scen[:act_fct], 
hl_weight=scen[:initializer_weight], hl_bias=scen[:initializer_bias]);

# train the model
my_NN, trainloss, testloss = train_model!(var_train_set, K_train_set, my_NN, lera=scen[:lera], beta=(scen[:beta1],scen[:beta2]), lambda=scen[:lambda],
batchsize=scen[:batchsize], nepochs=scen[:nepochs], early_stopping=scen[:early_stopping], adapt_lera=scen[:adapt_lera], lera_trend=scen[:lera_trend], x_test=var_test_set, y_test=K_test_set);

# plot learning curve
learning_curve = plot_losses(trainloss, testloss)
#savefig(learning_curve, String(savepath*"learning_curve_"*saveas*".png"))

# adds layers that include preprocessing
NN = reprocess_model(my_NN, var_prep_pars=var_prep_pars,K_prep_pars=K_prep_pars);
#save_model(NN, String(savepath*"NN_"*saveas*".jld2"))




# compare the kernel network prediction & interpolation
compare_kernels_ptur(var_set, K_func, NN, 0.143, 0.125, show_mse=true)
compare_kernels_temps(var_set, K_func, NN, 1.5, 1.5, show_mse=true)

# compare spectra calculated with network prediction & interpolation
compare_spectra_4D(0.143, 0.125, dic.pion, K_func, NN; pt_min=0., pt_max=3.7)





