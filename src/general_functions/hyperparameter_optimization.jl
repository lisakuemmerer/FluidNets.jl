


# THIS FUNCTION ONLY RANDOMLY TESTS GIVEN HYPERPARAMETERS !!!!
# IF YOU FIND A GOOD OPTIMIZATION PACKAGE IN JULA USE IT !!!!


# try the objective function trial() #num_trials times with parameters taken form scens
# scens needs to be Dictionary with hyperparameters & options
# hyperparameters need to be correctly unpacked in trial() as entered in scens
# excepts: remove impossible combinations. form: [p->p[:h1]==a && p[:h2]==b] will remove all combinations where scen[:h1]=a and scen[:h2]=b
function trials(scens; excepts=[], num_trials=100)
    
    t0 = time()

    #choose random option for each hyperparameter (only once), remove exeptions
    opts_to_try = unique([Dict{Symbol,Any}(k=>v[rand(eachindex(v))] for (k,v) in scens) for i in 1:num_trials])
    for e in excepts
        filter!(!(e), opts_to_try)
    end

    # set number of trials to run to number of options left after removing exceptions
    num_trials = length(opts_to_try)

    t1 = time()

    trial_vec = []
    Threads.@threads for i in 1:num_trials
        println("\n Starting trial ", i, "\n")
        push!(trial_vec, trial(opts_to_try[i]))
        println("Finished trial $(i); $(time()-t1)s")
    end
    
    println("Time for trials: $(time()-t1)s")
    println("Complete runtime: $(time()-t0)s")

    # return: array of dictionaries containing the hyperparameters and loss output of each option
    return trial_vec
end



# function to unpack readable (saveable) scenario dictionary
# scen_front should be dictionary containing the hyperparameter choices taken below
# if other options should be tested they need to be included here:

# preprocessing of variables, done on data called var_set, :prep_vars => ["none", "minwidth", "midhalfwidth", "meanstd"],
# preprocessing of kernel, done on data called K_set, :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax", "meanstd"], 
# activation function, "leakyrelu" results in leakyrelu default, strings in the form of "leakyrelu_01" will result in leakyrelu_grad(0.1), :act_fct => ["sigmoid","tanh", "relu", "leakyrelu", "leakyrelu_01"], 
# weights initializer, :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
# bias initializer, :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing", "zeros"],
# loss function, "xweight", "yweight" result in the defined default weights, otherwise the number after _ defines weight(>=1) and epsilon(<1), here: MyXWeightLoss(5), MyYweightLoss(0.01), :loss_fct => ["mse", "xweight", "yweight", "xweight_5", "yweight_001"], 

function scenario_frontend_to_backend(scen_front; var_set=var_set, K_set=K_set)

    # find all included keys
    scen_keys = keys(scen_front)

    # copy dictionary
    scen_back = Dict{Symbol, Any}(k => scen_front[k] for k in scen_keys)

    # unpack preprocessing parameters
    if :prep_vars in scen_keys
        if scen_front[:prep_vars] == "none"
            scen_back[:prep_vars] = false
        elseif scen_front[:prep_vars] == "minwidth"
            scen_back[:prep_vars] = get_min_width(var_set)
        elseif scen_front[:prep_vars] == "midhalfwidth"
            scen_back[:prep_vars] = get_mid_halfwidth(var_set)
        elseif scen_front[:prep_vars] == "meanstd"
            scen_back[:prep_vars] = get_mean_std(var_set)
        end
    end

    if :prep_K in scen_keys
        if scen_front[:prep_K] == "none"
            scen_back[:prep_K] = false
        elseif scen_front[:prep_K] == "minwidth"
            scen_back[:prep_K] = get_min_width(K_set)
        elseif scen_front[:prep_K] == "midhalfwidth"
            scen_back[:prep_K] = get_mid_halfwidth(K_set)
        elseif scen_front[:prep_K] == "zeroabsmax"
            scen_back[:prep_K] = get_zero_absmax(K_set)
        elseif scen_front[:prep_K] == "meanstd"
            scen_back[:prep_K] = get_mean_std(K_set)
        end
    end

    # unpack activation functions
    if :act_fct in scen_keys
        if scen_front[:act_fct] == "sigmoid"
            scen_back[:act_fct] = sigmoid_fast
        elseif scen_front[:act_fct] == "tanh"
            scen_back[:act_fct] = tanh_fast
        elseif scen_front[:act_fct] == "relu"
            scen_back[:act_fct]= relu
        elseif scen_front[:act_fct] == "leakyrelu"
            scen_back[:act_fct] = leakyrelu
        elseif occursin("leakyrelu_", scen_front[:act_fct])
            a = parse(Float64, String("0."*match(r"_0([0-9.]+)", scen_front[:act_fct]).captures[1]))
            scen_back[:act_fct] = leakyrelu_grad(a)
        end
    end

    # unpack initializers for weights in hidden layers
    if :initializer_weight in scen_keys
        if scen_front[:initializer_weight] == "glorot_normal"
            scen_back[:initializer_weight] = glorot_normal
        elseif scen_front[:initializer_weight] == "glorot_uniform"
            scen_back[:initializer_weight] = glorot_uniform
        elseif scen_front[:initializer_weight] == "kaiming_normal"
            scen_back[:initializer_weight] = kaiming_normal     
        elseif scen_front[:initializer_weight] == "kaiming_uniform"
            scen_back[:initializer_weight] = kaiming_uniform
        elseif scen_front[:initializer_weight] == "random_normal"
            scen_back[:initializer_weight] = randn32
        elseif scen_front[:initializer_weight] == "random_uniform"
            scen_back[:initializer_weight] = rand32
        elseif scen_front[:initializer_weight] == "nothing"
            scen_back[:initializer_weight] = nothing
        end
    end

    # unpack initializers for biases in hidden layers
    if :initializer_bias in scen_keys
        if scen_front[:initializer_bias] == "glorot_normal"
            scen_back[:initializer_bias] = glorot_normal
        elseif scen_front[:initializer_bias] == "glorot_uniform"
            scen_back[:initializer_bias] = glorot_uniform
        elseif scen_front[:initializer_bias] == "kaiming_normal"
            scen_back[:initializer_bias] = kaiming_normal
        elseif scen_front[:initializer_bias] == "kaiming_uniform"
            scen_back[:initializer_bias] = kaiming_uniform
        elseif scen_front[:initializer_bias] == "random_normal"
            scen_back[:initializer_bias] = randn32
        elseif scen_front[:initializer_bias] == "random_uniform"
            scen_back[:initializer_bias] = rand32
        elseif scen_front[:initializer_bias] == "zeros"
            scen_back[:initializer_bias] = zeros32
        elseif scen_front[:initializer_bias] == "nothing"
            scen_back[:initializer_bias] = nothing
        end
    end

    # unpack loss functions
    if :loss_fct in scen_keys
        if scen_front[:loss_fct] == "mse"
            scen_back[:loss_fct] = MSELoss()
        elseif scen_front[:loss_fct] == "xweight"
            scen_back[:loss_fct] = MyXweightLoss()
        elseif occursin("xweight_", scen_front[:loss_fct])
            w = parse(Float64, String(match(r"_([0-9.]+)", scen_front[:loss_fct]).captures[1]))
            scen_back[:loss_fct] = MyXweightLoss(w)
        elseif scen_front[:loss_fct] == "yweight"
            scen_back[:loss_fct] = MyYweightLoss()
        elseif occursin("yweight_", scen_front[:loss_fct])
            eps = parse(Float64, String("0."*match(r"_0([0-9.]+)", scen_front[:loss_fct]).captures[1]))
            scen_back[:loss_fct] = MyYweightLoss(eps)
        end
    end

    return scen_back
end




# function that runs & saves ALL given scenarios
# input: array of dictionaries containing hyperparameters, as output of trials
# if output from trials is given, number of hyperparameters needs to be corrected
# output: array of dictionaries containing hypperparameters & training parameters (trainstate, train&testloss, time of training...)
function run_all_trials(scen_vec; num_hyppars=nothing)

    t0 = time()
    
    # set number of hyperparameters if not less than all dict. entries
    num_hyppars === nothing && (num_hyppars=length(first(scen_vec)))
    
    # get keys of hyperparameters
    hyppars = [k for k in keys(first(scen_vec))][1:num_hyppars]

    t1 = time()

    trial_vec = []
    Threads.@threads for i in eachindex(scen_vec)
        #println("\n Starting trial ", i, "\n")
        push!(trial_vec, trial(scen_vec[i], deepsave=true))
        println("\n Finished trial ", i, "\n")
    end
    
    println("\n time for trials: ", time()-t1, "s")
    println("\n complete runtime: ", time()-t0, "s \n")

    # return: array of dictionaries containing the hyperparameters and loss output of each option
    return trial_vec
end




# save & load array of many small dics without trainstate
function save_trials(trialarray, filename)
    @save filename trialarray
end


function load_trials(filename)
 @load filename trialarray
    return trialarray
end




# merge trial files of directory to one file
function merge_trials(direct, filename)
    filelist = readdir(direct)
    trials = reduce(vcat, [load_trials(String(direct*f))[:] for f in filelist])
    save_trials(trials, direct*filename)
end


