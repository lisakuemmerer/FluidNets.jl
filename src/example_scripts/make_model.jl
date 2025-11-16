#include("/home/lisa/MA/NeuralNetwork/my_functions.jl");
#include("/home/lisa/MA/Fluidum/my_main.jl");

using FluidNets
using Lux




var_set, K_set = read_data("/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt", 4, 8);

K_func = extrapolate_interpolate_kernels(var_set, K_set);


#####################################################################################################################################################


var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set,
preprocess_vars=get_mean_std(var_set), preprocess_K=false, n_train=10000, n_test=10000);


my_NN = initiate_model(4, 8, nb_hl=6, hl_dim=256, act_fct=sigmoid_fast, 
hl_weight=glorot_uniform, hl_bias=randn32);



my_NN, trainloss, testloss = train_model!(var_train_set, K_train_set, my_NN, lera=0.00129155, beta=(0.99,0.9999),
batchsize=100, nepochs=10, early_stopping=true, x_test=var_test_set, y_test=K_test_set, loss_fct=MSELoss());


MY_NN = reprocess_model(my_NN, var_prep_pars=var_prep_pars,K_prep_pars=K_prep_pars);
#save_model(Trainstate, "/home/lisa/MA/NeuralNetwork/pion_4D_BG/NN.jld2")



#############



pl = plot_losses(trainloss, testloss)
#savefig(pl, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_learning_curve.png"))





################


ps = compare_spectra_PCE(0.12,0.14, dic.pion, K_func, MY_NN; decays=false, pt_min=0.016, pt_max=3.727, plotlog=true)
ps = compare_spectra_PCE(0.12,0.14, dic.pion, K_func, Trainstate; decays=false, pt_min=0.016, pt_max=0.2, steps=200, comp_ratio=false, plotlog=false)
#savefig(ps, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_spectra.png"))

pk = compare_kernels_PCE_ptur(var_set, K_func, my_NN, 0.143, 0.125; n=20)
pk = compare_kernels_PCE_temps(var_set, K_func, Trainstate, 1.5, 1.5; n=20)
#savefig(pk, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_kernels.png"))



#save_model(Trainstate, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_model.jld2"))

