

using FluidNets
using Fluidum
using Lux



# DATA

# load data containing 2 variables (pT,ur), 8 Kernels (Background)
# data: pion total spectrum, FO @ 156.5 MeV
var_set, K_set = read_data("/home/lisa/MA/Data/Pion2D/pion_total.txt", 2, 8);

# define interpolation of kernels to be able to compare prediction
K_func = extrapolate_interpolate_kernels(var_set, K_set);



# # OR generate random data:
# v1 = sort(rand(300)*(3.7261576-0.0161576) .+ 0.0161576)
# v2 = sort(rand(300)*3.5)
# var_set = FluidNets._get_set([v1,v2])

# p = [randn(3) for i in 1:8]
# K_func = [(x,y)->p[i][1]*x+p[i][2]*y+p[i][3] for i in eachindex(p)]
# K_set = hcat([[f(var_set[1,i], var_set[2,i]) for f in K_func] for i in axes(var_set, 2)]...)





# TRAINING

# separate data in train & test sets while saving preprocessing parameters
var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set, uniform=false);

# initialize model for 2 input variables, 8 output variables
my_NN = initiate_model(2, 8, nb_hl=4, hl_dim=32, act_fct=relu);

# train the model for 1000 epochs
my_NN, trainloss, testloss = train_model!(var_train_set, K_train_set, my_NN, 
x_test=var_test_set, y_test=K_test_set);

# plot learning curve
pl = plot_losses(trainloss, testloss)





# VERIFICATION

# add layers that include preprocessing for variables/Kernels so that NN can be used on data directly
NN = reprocess_model(my_NN, var_prep_pars=var_prep_pars,K_prep_pars=K_prep_pars);

# compare true function and network prediction
FluidNets.compare_kernels_2D(var_set, K_func, my_NN)



# compare spectra calculated with network prediction & interpolation
compare_spectra_2D(dic.pion, K_func, NN; pt_min=0., pt_max=3.7)


