


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


