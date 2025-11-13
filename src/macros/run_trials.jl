

# read in data
data = readdlm("/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt", Float64);

var_set = data[1:4,:];
K_set = data[5:end,:];

################################################################################################################

# run one training with given hyperparameters defined in scen
# hyperparameters need to be correctly unpacked or put in at correct argument
function trial(scen; deepsave=false)

    # unpack preprocessing parameters
    if scen[:prep_vars] == "none"
        pv = false
    elseif scen[:prep_vars] == "minwidth"
        pv = get_min_width(var_set)
    elseif scen[:prep_vars] == "midhalfwidth"
        pv = get_mid_halfwidth(var_set)
    end

    if scen[:prep_K] == "none"
        pk = false
    elseif scen[:prep_K] == "minwidth"
        pk = get_min_width(K_set)
    elseif scen[:prep_K] == "midhalfwidth"
        pk = get_mid_halfwidth(K_set)
    elseif scen[:prep_K] == "zeroabsmax"
        pk = get_zero_absmax(K_set)
    end

    # unpack activation functions
    if scen[:act_fct] == "sigmoid"
        actfct = sigmoid_fast
    elseif scen[:act_fct]  == "relu"
        actfct = relu
    elseif scen[:act_fct]  == "leakyrelu_01"
        actfct = leakyrelu
    elseif scen[:act_fct]  == "leakyrelu_02"
        actfct = leakyrelu_grad(0.2)
    end

    # unpack initializers for weights in hidden layers
    if scen[:initializer_weight] == "glorot_normal"
        initializer_weight = initializer_gain(glorot_normal, actfct)
    elseif scen[:initializer_weight] == "glorot_uniform"
        initializer_weight = initializer_gain(glorot_uniform, actfct)
    elseif scen[:initializer_weight] == "kaiming_normal"
        initializer_weight = initializer_gain(kaiming_normal, actfct)      
    elseif scen[:initializer_weight] == "kaiming_uniform"
        initializer_weight = initializer_gain(kaiming_uniform, actfct)
    elseif scen[:initializer_weight] == "random_normal"
        initializer_weight = randn32
    elseif scen[:initializer_weight] == "random_uniform"
        initializer_weight = rand32
    elseif scen[:initializer_weight] == "nothing"
        initializer_weight = nothing
    end

    # unpack initializers for biases in hidden layers
    if scen[:initializer_bias] == "glorot_normal"
        initializer_bias = initializer_gain(glorot_normal, actfct)
    elseif scen[:initializer_bias] == "glorot_uniform"
        initializer_bias = initializer_gain(glorot_uniform, actfct)
    elseif scen[:initializer_bias] == "kaiming_normal"
        initializer_bias = initializer_gain(kaiming_normal, actfct)      
    elseif scen[:initializer_bias] == "kaiming_uniform"
        initializer_bias = initializer_gain(kaiming_uniform, actfct)
    elseif scen[:initializer_bias] == "random_normal"
        initializer_bias = randn32
    elseif scen[:initializer_bias] == "random_uniform"
        initializer_bias = rand32
    elseif scen[:initializer_bias] == "nothing"
        initializer_bias = nothing
    end

    # unpack loss functions
    if scen[:loss_fct] == "mse"
        lossfct = MSELoss()
    elseif scen[:loss_fct] == "xweight"
        lossfct = MyXweightLoss()
    elseif scen[:loss_fct] == "yweight"
        lossfct = MyYweightLoss()
    end


    var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set,
    preprocess_vars=pv, preprocess_K=pk, n_train=10000, n_test=10000);
    my_NN = initiate_model(4, 8, nb_hl=scen[:nb_hl], hl_dim=scen[:hl_dim], act_fct=actfct, hl_weight=initializer_weight, hl_bias=initializer_bias);
    my_NN, trainloss, testloss, tft, overfit = train_model!(var_train_set, K_train_set, my_NN, 
    batchsize=scen[:batchsize], loss_fct=lossfct, lera=scen[:lera], beta=(scen[:beta1],scen[:beta2]), lambda=scen[:lambda], 
    nepochs=1000, x_test=var_test_set, y_test=K_test_set, optim_mode=true, messages=false);

    # return: dictionary containing used hyperparameters, time for training, testloss, and model_overfit
    dict = OrderedDict{Symbol,Any}(k=>v for (k,v) in scen)
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
# if you want to evaluate different hyperparameters you need to adjust the trial function accordingly
scens = Dict{Symbol, Any}(:prep_vars => ["none", "minwidth", "midhalfwidth"],
    :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax"], 
    :nb_hl => [4,5,6,7], 
    :hl_dim => [32,64,128,264], 
    :act_fct => ["sigmoid", "relu", "leakyrelu_01", "leakyrelu_02"], 
    :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
    :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
    :batchsize => [100,500,1000,5000], 
    :loss_fct => ["mse", "xweight", "yweight"], 
    :lera => 10 .^LinRange(-4,-2,10),
    :beta1 => [0.8,0.85,0.9,0.95,0.99], 
    :beta2 => [0.99,0.999,0.9999], 
    :lambda => [0.0,0.01,0.02]);


# put in exeptions for combinations of parameter choices that do not make sense
excepts = [s->s[:loss_fct]=="yweight" && (s[:prep_K]=="midhalfwidth" || s[:prep_K]=="minwidth"), 
    s -> s[:loss_fct]=="xweight" && s[:prep_vars]=="none"];


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



# run trials. tis take a long time!
# function will call trial
trial_all = trials(scens, excepts=excepts, num_trials=2);


# save the trials
#save_trials(trial_all, "/home/lisa/MA/NeuralNetwork/hyperparameter/testtrial.jld2");

