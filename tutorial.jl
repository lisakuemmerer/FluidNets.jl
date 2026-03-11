

using FluidNets
using Fluidum
using Lux


# DATA

# load data containing 4 variables (pT,ur,Tchem,Tkin), 8 Kernels (Background)
var_set, K_set = read_data("/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt", 4, 8);



# TRAINING

# separate data in train & test sets while saving preprocessing parameters
# preprocessing options chosen here: normalize variables, no preprocessing for kernels
# number of samples in train & test set: 10k each
var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set,
preprocess_vars=get_mean_std(var_set), preprocess_K=false, n_train=10000, n_test=10000);

# initialize model for 4 input variables, 8 output variables, 6 hidden layer with 256 nodes each
# activation function sigmoid & custom weight, bias initialization in hidden layers
my_NN = initiate_model(4, 8, nb_hl=6, hl_dim=256, act_fct=sigmoid_fast, 
hl_weight=glorot_uniform, hl_bias=randn32);

# train the model for 1000 loops
my_NN, trainloss, testloss = train_model!(var_train_set, K_train_set, my_NN, lera=0.001, beta=(0.9,0.999),
batchsize=500, nepochs=1000, x_test=var_test_set, y_test=K_test_set, loss_fct=MSELoss());

# plot learning curve
pl = plot_losses(trainloss, testloss)



# VERIFICATION

# add layers that include preprocessing for variables/Kernels so that NN can be used on data directly
NN = reprocess_model(my_NN, var_prep_pars=var_prep_pars,K_prep_pars=K_prep_pars);

# define interpolation of kernels to be able to compare prediction
K_func = extrapolate_interpolate_kernels(var_set, K_set);

# compare the kernels (network prediction & interpolation)
# compare in pt,ur for Tchem=0.143, Tkin=0.125
compare_kernels_ptur(var_set, K_func, NN, 0.143, 0.125)
# compare in tchem,Tkin for pt=ur=1.5
compare_kernels_temps(var_set, K_func, NN, 1.5, 1.5)

# compare spectra calculated with network prediction & interpolation for Tchem=0.143, Tkin=0.125
compare_spectra_4D(0.143, 0.125, dic.pion, K_func, NN; pt_min=0., pt_max=3.7)




