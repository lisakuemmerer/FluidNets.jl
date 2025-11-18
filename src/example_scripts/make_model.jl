

using FluidNets
using Fluidum
using Lux


# load data
var_set, K_set = read_data("/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt", 4, 8);

# define interpolation of kernels to be able to compare prediction
K_func = extrapolate_interpolate_kernels(var_set, K_set);



# separate data in train & test sets, save preprocessing parameters
var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set,
preprocess_vars=get_mean_std(var_set), preprocess_K=false, n_train=10000, n_test=10000);

# initialize model
my_NN = initiate_model(4, 8, nb_hl=6, hl_dim=256, act_fct=sigmoid_fast, 
hl_weight=glorot_uniform, hl_bias=randn32);

# train the model
my_NN, trainloss, testloss = train_model!(var_train_set, K_train_set, my_NN, lera=0.00129155, beta=(0.99,0.9999),
batchsize=100, nepochs=10, early_stopping=true, x_test=var_test_set, y_test=K_test_set, loss_fct=MSELoss());

# plot learning curve
pl = plot_losses(trainloss, testloss)

# adds layers that include preprocessing
NN = reprocess_model(my_NN, var_prep_pars=var_prep_pars,K_prep_pars=K_prep_pars);





# # alternative approach: load saved model
# NN = load_model("/home/lisa/MA/NeuralNetwork/pion_4D_BG/NN.jld2");



# compare the kernel network prediction & interpolation
compare_kernels_ptur(var_set, K_func, NN, 0.143, 0.125)
compare_kernels_temps(var_set, K_func, NN, 1.5, 1.5)

# compare spectra calculated with network prediction & interpolation
compare_spectra_4D(0.143, 0.125, dic.pion, K_func, NN; pt_min=0., pt_max=3.7)





