

##############################################################################################################
# This script allows to run hyperparameter optimization.
# the 'trial' function will be called for different options defined below by the 'trials' function



# read in data
var_set, K_set = read_data("/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt", 4, 8);


################################################################################################################

# run one training with given hyperparameters defined in scen
# hyperparameters need to be correctly unpacked or put in at correct argument
# this function can be customized according to which hyperparameters should be tested
function trial(scen_frt; deepsave=false)

    scen = scenario_frontend_to_backend(scen_frt)

    var_train_set, K_train_set, var_test_set, K_test_set, _, _ = get_train_test_set(var_set, K_set,
    preprocess_vars=scen[:prep_vars], preprocess_K=scen[:prep_K], n_train=10000, n_test=10000);
    my_NN = initiate_model(4, 8, nb_hl=scen[:nb_hl], hl_dim=scen[:hl_dim], act_fct=scen[:act_fct], hl_weight=scen[:initializer_weight], hl_bias=scen[:initializer_bias]);
    my_NN, trainloss, testloss, tft, overfit = train_model!(var_train_set, K_train_set, my_NN, 
    batchsize=scen[:batchsize], loss_fct=scen[:loss_fct], lera=scen[:lera], beta=(scen[:beta1],scen[:beta2]), lambda=scen[:lambda], 
    nepochs=1000, x_test=var_test_set, y_test=K_test_set, optim_mode=true, messages=false);

    # return: dictionary containing used hyperparameters, time for training, testloss, and model_overfit
    dict = OrderedDict{Symbol,Any}(k=>v for (k,v) in scen_frt)
    !(deepsave) && (dict[:endloss] = testloss[2][end])
    !(deepsave) && (dict[:improv] = testloss[2][end]/testloss[2][1])
    deepsave && (dict[:NN] = my_NN)
    deepsave && (dict[:trainloss] = trainloss)
    deepsave && (dict[:testloss] = testloss)
    dict[:tft] = tft
    dict[:overfit] = overfit

    return dict
end


# put in hyperparameter options you wish to try
# function can not be saved correctly, which is why strings are used @ frontend. options are explained in scenario_frontend_to_backend and can be adjusted there
# if you want to evaluate different hyperparameters you need to adjust the trial function accordingly
scens = Dict{Symbol, Any}(:prep_vars => ["none", "minwidth", "midhalfwidth", "meanstd"],
    :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax", "meanstd"], 
    :nb_hl => [4,5,6,7], 
    :hl_dim => [32,64,128,264], 
    :act_fct => ["sigmoid","tanh", "relu", "leakyrelu_001", "leakyrelu_01"], 
    :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
    :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing", "zeros"],
    :batchsize => [100,500,1000,5000], 
    :loss_fct => ["mse", "xweight", "yweight"], 
    :lera => 10 .^LinRange(-4,-2,10),
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



# # DO NOT UNCOMMENT UNLESS WANTED, RUNS FOR MULTIPLE HOURS !!!
# # function will call trial-function and run #num_trials options from scen without excepts
#trial_all = trials(scens, excepts=excepts, num_trials=2);


# save the trials
#save_trials(trial_all, "/home/lisa/MA/NeuralNetwork/hyperparameter/testtrial.jld2");

export trial, scen, excepts

