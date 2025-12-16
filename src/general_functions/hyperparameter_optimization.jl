

##############################################################################################################
# run different scenarios 
# THESE FUNCTIONs ONLY RANDOMLY TESTS GIVEN HYPERPARAMETERS !!!!
# IF YOU FIND A GOOD OPTIMIZATION PACKAGE IN JULA USE IT !!!!




# function to unpack readable (saveable) scenario dictionary
# scen_front should be dictionary containing the hyperparameter choices taken below
# if other options should be tested they need to be included here:

# preprocessing of variables, done on data called var_set, :prep_vars => ["none", "minwidth", "midhalfwidth", "meanstd"],
# preprocessing of kernel, done on data called K_set, :prep_K => ["none", "minwidth", "midhalfwidth", "zeroabsmax", "meanstd"], 
# activation function, "leakyrelu" results in leakyrelu default, strings in the form of "leakyrelu_01" will result in leakyrelu_grad(0.1), :act_fct => ["sigmoid","tanh", "relu", "leakyrelu", "leakyrelu_01"], 
# weights initializer, :initializer_weight => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing"],
# bias initializer, :initializer_bias => ["glorot_normal", "glorot_uniform", "kaiming_normal", "kaiming_uniform", "random_normal", "random_uniform", "nothing", "zeros"],
# loss function, "xweight", "yweight" result in the defined default weights, otherwise the number after _ defines weight(>=1) and epsilon(<1), here: MyXWeightLoss(5), MyYweightLoss(0.01), :loss_fct => ["mse", "xweight", "yweight", "xweight_5", "yweight_001"], 

function scenario_frontend_to_backend(scen_front, var_set, K_set)

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





# run one training with given hyperparameters defined in scen
# hyperparameters need to be correctly unpacked or put in at correct argument
# this function can be customized according to which hyperparameters should be tested
function _trial(scen_frt, var_set, K_set; deepsave=false)
    var_dim = size(var_set, 1)
    K_dim = size(K_set, 1)

    scen = scenario_frontend_to_backend(scen_frt, var_set, K_set)

    var_train_set, K_train_set, var_test_set, K_test_set, _, _ = get_train_test_set(var_set, K_set,
    preprocess_vars=scen[:prep_vars], preprocess_K=scen[:prep_K], n_train=10000, n_test=10000);
    my_NN = initiate_model(var_dim, K_dim, nb_hl=scen[:nb_hl], hl_dim=scen[:hl_dim], act_fct=scen[:act_fct], hl_weight=scen[:initializer_weight], hl_bias=scen[:initializer_bias]);
    my_NN, trainloss, testloss, tft, overfit = train_model!(var_train_set, K_train_set, my_NN, 
    batchsize=scen[:batchsize], loss_fct=MSELoss(), lera=scen[:lera], beta=(scen[:beta1],scen[:beta2]), lambda=scen[:lambda], 
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



# helper function to have nicer time formats
function _format_seconds(seconds)
    h = div(seconds, 3600)
    m = div(rem(seconds, 3600), 60)
    s = rem(seconds, 60)
    return string(lpad(Int(h), 2, "0"), ":", lpad(Int(m), 2, "0"), ":", lpad(round(Int, s), 2, "0"))
end



# try the objective function _trial() #num_trials times with parameters taken form scens
# scens needs to be Dictionary with hyperparameters & options
# hyperparameters need to be correctly unpacked in trial() as entered in scens
# excepts: remove impossible combinations. form: [p->p[:h1]==a && p[:h2]==b] will remove all combinations where scen[:h1]=a and scen[:h2]=b
function trials(scens, var_set, K_set; excepts=[], num_trials=100, multithread=true)
    
    t0 = time()

    #choose random option for each hyperparameter (only once), remove exeptions
    opts_to_try = unique([Dict{Symbol,Any}(k=>v[rand(eachindex(v))] for (k,v) in scens) for i in 1:num_trials])
    for e in excepts
        filter!(!(e), opts_to_try)
    end

    # set number of trials to run to number of options left after removing exceptions
    num_trials = length(opts_to_try)

    t1 = time()

    trial_vec = Vector{Any}(undef, num_trials)
    if multithread
        thread_task_counts = zeros(Int, Threads.nthreads())
        Threads.@threads for i in 1:num_trials
            tid = Threads.threadid()-1
            thread_task_counts[tid] += 1
            local_task_num = thread_task_counts[tid]
            println("Starting trial ", tid, ".", local_task_num)
            trial_vec[i] = _trial(opts_to_try[i], var_set, K_set)
            println("Finished trial $(tid).$(local_task_num); $(_format_seconds(time()-t1))")
        end
    else
        for i in 1:num_trials
            println("Starting trial ", i)
            trial_vec[i] = _trial(opts_to_try[i], var_set, K_set)
            println("Finished trial $(i); $(_format_seconds(time()-t1))")
        end
    end
    
    println("Time for trials: $(_format_seconds(time()-t1))")
    println("Complete runtime: $(_format_seconds(time()-t0))")

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




# merge trial files of directory to one file with option of removing double entries
function merge_trials(direct, filename; hyppars=nothing)
    filelist = readdir(direct)
    trials = reduce(vcat, [load_trials(String(direct*f))[:] for f in filelist])
    hyppars!==nothing && (trials=unique(d -> Tuple(d[k] for k in hyppars), sort(trials, by = x->x[:endloss])))
    save_trials(trials, direct*filename)
end




# funtion to save one hyperparameter choice readable
function save_hyppars(hyppars, outfile)
    open(outfile, "w") do f
        println(f, "Dict{Symbol, Any}(")
        for (k,v) in hyppars
            Pair{Symbol, Any}(k,v) != first(hyppars) && (print(f, ", \n"))
            if v isa String
                print(f, "$k => \"$v\"")
            else
                print(f, "$k => $v")
            end
        end
        println(f, ")")
    end
end





##############################################################################################################
# evaluate different scenarios


# sort trials by result
sortby(Trials; verific=:endloss) = sort(Trials, by=t->t[verific])


# get all the included hyperparameter options
function get_options(Trials, hyppars; verific=:endloss)

    # accomodate option of only one hyperparameter
    hyppars isa Symbol && (hyppars = [hyppars])

    # fill dictionary with empty array for each hyperparameter
    scen = OrderedDict{Symbol, Any}(h=>[] for h in hyppars)
    # fill dictionary entries with sorted options for each hyperparameter
    for h in hyppars
        for t in Trials
            t[h] in scen[h] ? nothing : push!(scen[h], t[h])
        end
        if first(scen[h]) isa String && verific in keys(first(Trials))
            sort!(scen[h], by=v->mean(t[verific] for t in filter(t->t[h]==v, Trials)))
        else
            sort!(scen[h])
        end
    end

    # return only array of options if one hyperparameter is wanted
    length(scen) == 1 && (scen = first(values(scen)))

    return scen
end

# function to make a histogram of of endloss for different hyperparameter choices
function hist_loss(hyppar, Trials; trialmax=nothing, verific=:endloss)
    trialmax===nothing && (trialmax=length(Trials))
    t = sortby(Trials, verific=verific)[1:trialmax]
    vs = get_options(t, hyppar)
    l_max = t[end][verific]
    l_min = t[1][verific]
    h = [histogram([t[verific] for t in filter(t->t[hyppar]==v,t)], label="", bins=10, title=v, xlims=(l_min, l_max)) for v in vs]
    return plot(h..., ylims=(0,maximum([ylims(h)[2] for h in h])), plot_title="$(String(verific)) occurence for $(String(hyppar))")
end


# function to get
function get_count_in_best_trials(h, Trials; trialmax=nothing, verific=:endloss)
    trialmax===nothing && (trialmax=length(Trials))
    return countmap([t[h] for t in sortby(Trials, verific=verific)[1:trialmax]])
end


# function to get mean result of hyperparameter choice
function get_mean_result(h, Trials; trialmax=nothing, verific=:endloss)
    trialmax===nothing && (trialmax=length(Trials))
    Dict{Any,Any}(v=>mean(t[verific] for t in filter(t->t[h]==v, sortby(Trials, verific=verific)[1:trialmax])) for v in get_options(Trials,h))
end


# function to plot one hyperparameter with result
function plot_hyppar(hyppar, Trials; trialmax=nothing, xscale=:identity, yscale=:log10, verific=:endloss)
    trialmax===nothing && (trialmax=length(Trials))
    Trials = sortby(Trials, verific=verific)[1:trialmax]
    plot([t[hyppar] for t in Trials], [t[verific] for t in Trials], st=:scatter, xscale=xscale, yscale=yscale, title=String(hyppar), ylabel=String(verific), label="")
end


# function to plot the course of how often a hyperparameterchoice makes it to the best trials
function plot_course_in_best_trials(hyppar, Trials; trialmax=nothing, verific=:endloss)
    trialmax===nothing && (trialmax=length(Trials))
    # sort trials
    Trials_sorted = sortby(Trials, verific=verific)
    # get hyperparameter options
    vs = get_options(Trials, hyppar, verific=verific)
    # fill a dictionary with functions counting how often the options occur in best ... trials
    d = Dict{Any, Any}(v=>n->count(t->t[hyppar]==v, Trials_sorted[1:n]) for v in vs)
    
    # define range for plot
    n = 0:10:trialmax
    # make sure colours are in order 
    series_colors = reshape([cgrad(:viridis)[z] for z in range(0, 1, length=length(vs))], 1, length(vs))

    p = return plot(n, [d[v].(n)/d[v](length(Trials))*100 for v in vs], labels=reduce(hcat,vs), seriescolor = series_colors,
    xlabel="number of best trials", ylabel="% in best trials", title=String(hyppar))

    return p
end


# function that plots above plots for all hyppars at once
function plot_all_hyppars(f, hyppars, Trials; kwargs...)
    # number of hyppars -> size of plot
    n = div(length(hyppars), 3, RoundUp)
    
    # extract plottitle if given
    title = get(kwargs, :title, "") 

    # make tuple out of remaining kwargs
    inner_kwargs = NamedTuple(p for p in kwargs if p.first != :title)

    # plot all hyperparameters
    plot([f(h, Trials; inner_kwargs...) for h in hyppars]..., layout=(n, 3), size=(n * 500, 1800), left_margin=10*Plots.mm, bottom_margin=10*Plots.mm, plot_title=title)
end


# function to plot the correlation of two hyperparameters h1,h2
function plot_correlation(h1, h2, Trials; trialmax=nothing, verific=:endloss, xscale=:identity, yscale=:identity, log_res=true)

    trialmax===nothing && (trialmax=length(Trials))
    Trials=sortby(Trials, verific=verific)[1:trialmax]

    
    #get included options for hyperparameters
    v1,v2 = values(get_options(Trials, [h1,h2], verific=verific))

    #fill matrix with mean endloss for combination
    Comb = fill(NaN, length(v2), length(v1))
    for i in eachindex(v2)
        for j in eachindex(v1)
            filtered = filter(t->t[h1]==v1[j]&&t[h2]==v2[i], Trials)
            Comb[i,j] = isempty(filtered) ? NaN : mean(t[verific] for t in filtered)
        end
    end

    #plot with corresponding options
    if log_res
        p =  heatmap(v1,v2, log10.(Comb), xscale=xscale, yscale=yscale, 
        xlabel=String(h1), ylabel=String(h2), colorbar_title="$(String(verific)) (logarithmic)", c=:viridis)
    else
        p = heatmap(v1,v2, Comb, xscale=xscale, yscale=yscale, 
        xlabel=String(h1), ylabel=String(h2), colorbar_title=String(verific), c=:viridis)
    end

    return p
end




# function that histogramms occurance of hyperparameter choice in dataset (makes sense to use on ...best/worst trials)
function hist_occurance(hyppar, Trials; verific=:endloss, trialmax=nothing)
    trialmax===nothing && (trialmax=length(Trials))
    counts = countmap([t[hyppar] for t in sortby(Trials, verific=verific)[1:trialmax]])
    bar(collect(keys(counts)), collect(values(counts)),title="Frequency",ylabel="Count",legend=false)
end




# function that histogramms occurance of two hyperparameter choices in dataset (makes sense to use on ...best/worst trials)
function hist_correlation_occurance(h1, h2, Trials; verific=:endloss, trialmax=nothing)
    trialmax===nothing && (trialmax=length(Trials))
    Trials = sortby(Trials, verific=verific)[1:trialmax]

    # Get all unique values for both hyperparameters
    v1 = unique([t[h1] for t in Trials])
    v2 = unique([t[h2] for t in Trials])
    
    # Create a matrix to store counts
    counts = zeros(Int, length(v2), length(v1))
    
    # Fill the matrix
    for (i, val2) in enumerate(v2)
        for (j, val1) in enumerate(v1)
            counts[i, j] = count(t -> t[h1] == val1 && t[h2] == val2, Trials)
        end
    end
    
    # Plot heatmap
    heatmap(string.(v1), string.(v2), counts, 
            xlabel=String(h1), ylabel=String(h2), 
            title="Co-occurrence Frequency", c=:viridis)
end





##############################################################################################################
# OTHER STUFF: test scenarios deeper ( might not work anymore ? )

# function that runs & saves ALL given scenarios - use if you already have some good options that you want to compare deeper
# input: array of dictionaries containing hyperparameters, as output of trials
# if output from trials is given, number of hyperparameters needs to be corrected
# output: array of dictionaries containing hypperparameters & training parameters (trainstate, train&testloss, time of training...)
function run_all_trials(scen_vec; hyppars=nothing)

    t0 = time()
    
    # remove  possible result entries in scen_vec dicts
    hyppars!=nothing && (scen_vec=[Dict{Symbol, Any}((h=>s[h] for h in hyppars)) for s in scen_vec])

    t1 = time()

    trial_vec = []
    Threads.@threads for i in eachindex(scen_vec)
        #println("\n Starting trial ", i, "\n")
        push!(trial_vec, _trial(scen_vec[i], deepsave=true))
        println("\n Finished trial ", i, "\n")
    end
    
    println("\n time for trials: ", _format_seconds(time()-t1))
    println("\n complete runtime: ", _format_seconds(time()-t0), " \n")

    # return: array of dictionaries containing the hyperparameters and loss output of each option
    return trial_vec
end