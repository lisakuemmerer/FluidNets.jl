


abstract type own_model_structures end
struct model_structure{}<:own_model_structures
    model::Any
    parameters::Any
    states::Any
end
model_structures = Union{Lux.Training.TrainState, own_model_structures}



# if replicatability is wanted plant seed, use Lux.replicate(rng) always to get same seed & not mutate
def_rng = Random.default_rng() 
#Random.seed!(rng, 0)







########## MODEL ##################################################################################################################


#custom Loss functions
MyLogLoss = GenericLossFunction((ŷ, y) -> (log(abs(ŷ - y)+1)));
MyM3ELoss = GenericLossFunction((ŷ, y) -> abs.(ŷ - y)^3);
MyExpLoss = GenericLossFunction((ŷ, y) -> exp(abs(ŷ - y))-1);
MyCELoss = GenericLossFunction((ŷ, y) -> -y*log(ŷ) - (1-y)*log(1-ŷ)); # use only with out_act_fct=sigmoid or other scaled output
MyYweightLoss(epsilon) = GenericLossFunction((ŷ, y) -> abs2(ŷ - y) / (abs(y) + epsilon)); # use only on unprocessed Kernels

# puts weigth on high pT; weight on high&low pT when used on get_mean_width(var_set). 
# can be changed to put weight on other parameters as well
function MyXweightLoss(m, p, s, (x, y_true), w)
    y_pred, st = m(x, p, s)
    weight = w*abs.(x[1,:]) .+1
    weight = reshape(weight,1,:)
    loss = mean(weight .* (y_pred .- y_true).^2)
    return loss, st, (;)
end
MyXweightLoss(w) = (m, p, s, (x, y_true)) -> MyXweightLoss(m, p, s, (x, y_true), w)
# function MyXweightLoss(y_pred, y_true)
#     loss = mean((y_pred .- y_true).^2)
#     return loss
# end



# custom leakyrelu function with differing slope, =relu for a=0
leakyrelu_grad(a) = (b->leakyrelu(b,a))

#############################################################################

function gain(act_fct)
    act_fct == relu && (gain = sqrt(2))
    act_fct == identity && (gain = 1)
    act_fct == sigmoid_fast && (gain = 1)
    act_fct == leakyrelu && (gain = sqrt(2/(1+0.1^2)))
    act_fct == leakyrelu_grad(0.2) && (gain = sqrt(2/(1+0.2^2)))
    return gain
end


function initializer_gain(initializer, act_fct)
    init_with_gain(rng, dims...) = initializer(rng, dims...; gain=gain(act_fct))
    return init_with_gain
end 



# initiate model for inp_dim=number of variables (input), out_dim=number of kernels (output)
# possible arguments: nb_hl=number of hidden layers, hl_dim=number of neurons in each hidden layer, 
# act_fct=activation function for hidden layers, out_act_fct=activation function for output layer, 
# rng=random number generator, param64=convert parameters to Float64, 
# hl_weight, hl_bias, inp_weight, inp_bias, out_weight, out_bias=initialization of weights and biases in respective layers
function initiate_model(inp_dim, out_dim; nb_hl=5, hl_dim=32, act_fct=relu, in_act_fct=identity, out_act_fct=identity, rng=def_rng, param64=true,
                    hl_weight=nothing, hl_bias=nothing, inp_weight=nothing, inp_bias=nothing, out_weight=nothing, out_bias=nothing)

    # make chain of hidden layers
    hidden_layers = []
    for i in 1:nb_hl
        push!(hidden_layers, Dense(hl_dim=>hl_dim, act_fct, init_weight=hl_weight, init_bias=hl_bias))
    end
    # model: input layer, hidden layers, output layer
    model = Chain(Dense(inp_dim=>hl_dim, in_act_fct, init_weight=inp_weight, init_bias=inp_bias), 
    hidden_layers, Dense(hl_dim=>out_dim, out_act_fct, init_weight=out_weight, init_bias=out_bias))

    # initialize random parameters & empty states
    params, states = Lux.setup(rng,model)

    # change parameters to Float64 if param64 is true
    param64 && (params=Functors.fmap(x -> x isa AbstractArray ? Float64.(x) : x, params))

    # return model: input layer, hidden layers, output layers in form of Lux.Chain
    # return params: random initiazlied parameters in form of Lux.Parameters
    # return states: empty states in form of Lux.States
    return model_structure(model, params, states)
end

function batch_data(x,y,batchsize)
    
    shuffled_inds = shuffle(axes(x,2))

    shuffled_x = x[:,shuffled_inds][:,1:size(x,2)-size(x,2)%batchsize]
    shuffled_y = y[:,shuffled_inds][:,1:size(y,2)-size(y,2)%batchsize]

    reshaped_shuffled_x = reshape(shuffled_x, size(x,1),batchsize,:)
    reshaped_shuffled_y = reshape(shuffled_y, size(y,1),batchsize,:)
    D = [(reshaped_shuffled_x[:,:,i], reshaped_shuffled_y[:,:,i]) for i in axes(reshaped_shuffled_x, 3)]

    return D
end


# train model on dataset x=variables, y=true Kernels, model, parameters, states as initialized by initiate_model
# x_test, y_test = independent test set, batchsize=number of datapoints in one batch, nepochs=number of epochs to train,
# update_step=number of epochs after which test loss is calculated, lera=learning rate, beta,lambda=AdamW parameters,
# adapt_lera=true if learning rate should be adapted, lera_update_step=number of epochs after which learning rate is updated,
# lera_trend=trend of learning rate (0.999 for 0.1% decrease; if lera_trend is a tuple, it is interpreted as (end, number of steps) and the learning rate is linearly interpolated between input lera, lera_trend[1] with lera_trend[2] steps. after this the last value lera_trend[1] is used until training ends
# loss_fct=loss function to use, early_stopping=true if training should stop if test loss > train loss increases for 10 epochs, messages=true if messages should be printed
function train_model!(x, y, NN_init; x_test=x, y_test=y, batchsize=500, nepochs=1000, 
    update_step=10, lera=0.001, beta=(0.9,0.999), lambda=0., 
    adapt_lera=false, lera_update_step=10, lera_trend=0.999,
    loss_fct=MSELoss(), early_stopping=true, messages=true, optim_mode=false)

    #build leraning rate vector if lera_trend is a tuple
    if adapt_lera==true && lera_trend isa Tuple
        lera_vector = reverse(LinRange(lera_trend[1], lera, lera_trend[2]))
    end

    # build trainstate according to Lux training
    #DatLoad = DataLoader((x,y), batchsize=batchsize, partial=false, shuffle=true) # do not use on Slurm, f***s up everything
    train_state = Training.TrainState(NN_init.model, NN_init.parameters, NN_init.states, AdamW(lera,beta,lambda))
    
    # calculate initial loss
    #K0, _ = Lux.apply(train_state.model, x, train_state.parameters, train_state.states)
    #in_loss = loss_fct(K0,y)
    in_loss,_,_ = loss_fct(train_state.model, train_state.parameters, train_state.states, (x, y))

    messages && println("\n initial loss ", in_loss)


    # initialize needed output arguments
    train_loss = ([],[])
    test_loss = ([],[])
    epoch_test_loss = 0.
    counter = 0
    overfit = false
    t0 = time()


    for i in 1:nepochs

        # train step: update on every batch in dataloader, batch loss is output 2
        # mean over whole dataset, append epoch & trainloss
        epoch_train_loss = mean([Training.single_train_step!(AutoZygote(), loss_fct, (xbatch, ybatch), train_state)[2] for (xbatch, ybatch) in batch_data(x,y,batchsize)])
        push!(train_loss[2], epoch_train_loss)
        push!(train_loss[1], i)

        # test for fuck up
        if isnan(epoch_train_loss) 
            println("model broke")
            break
        end

        # test on testset
        if i%update_step == 0
            #Km, _ = Lux.apply(train_state.model, x_test, train_state.parameters, train_state.states)
            #epoch_test_loss = loss_fct(Km,y_test)
            epoch_test_loss,_,_ = loss_fct(train_state.model, train_state.parameters, train_state.states, (x_test, y_test))
            push!(test_loss[2], epoch_test_loss)
            push!(test_loss[1], i+1)
            t = time()
            messages && println("after ", i, " epochs: time ", t-t0, " s, trainloss: ", epoch_train_loss, ", testloss (updated): ", epoch_test_loss)
        end
        
        # compare testloss/trainloss for early stopping
        if i%update_step == 1
            dep_rat = epoch_test_loss/epoch_train_loss
            if dep_rat > 1.
                counter += 1
            else
                counter = 0
            end
        end


        # stop if testloss > trainloss for 10 successive epochs
        if early_stopping && counter >= 10
            messages && println("model overfitting")
            overfit = true
            break
        end

        # update learning rate if adapt_lera is true
        # the trainstate is reinitialized with empty optimizer state -> momentum lost
        # THIS SHOULD BE CHANGED ONCE LUX ALLOWS TO UPDATE THE TRAINSTATE!!!!!!!!!!!!
        if adapt_lera && i%lera_update_step==0 && i!=nepochs
            if lera_trend isa Number
                eta = lera_trend*train_state.optimizer.eta
                messages && println("updating learning rate")
                optimizer = AdamW(eta, beta, lambda)
                train_state = Training.TrainState(train_state.model, train_state.parameters, train_state.states, optimizer)
            elseif lera_trend isa Tuple && i / lera_update_step + 1 <= length(lera_vector)
                eta = lera_vector[Int(i / lera_update_step) + 1 ]
                messages && println("updating learning rate")
                optimizer = AdamW(eta, beta, lambda)
                train_state = Training.TrainState(train_state.model, train_state.parameters, train_state.states, optimizer)
            end
        end
        
    end


    # measure time needed for training
    t1 = time()
    tft = t1-t0

    # calculate final loss
    #Kf, _ = Lux.apply(train_state.model, x, train_state.parameters, train_state.states)
    #out_loss = loss_fct(Kf,y)
    out_loss,_,_ = loss_fct(train_state.model, train_state.parameters, train_state.states, (x, y))
    messages && println("Time for training: ",  tft, "s\n initial loss: ", in_loss, " final loss: ", out_loss, " Improv: ", out_loss/in_loss, "\n")
    
    # return NN structure: updated model with trained parameters
    # return trainloss: tuple with (epoch, loss) for training loss (epoch & loss as array)
    # return testloss: tuple with (epoch, loss) for test loss (epoch & loss as array)
    # return time for training tft

    NN_trained = model_structure(train_state.model, train_state.parameters, train_state.states)
    optim_mode ? ret = (NN_trained, train_loss, test_loss, tft, overfit) : ret = (NN_trained, train_loss, test_loss)

    return ret
end


# #train model with custom learning rate (vector) !!!! careful: makes nepochs*length(lera) updates !!!!
# lera = reverse([r for r in 1e-4:1e-4:1e-3])
# l = (x -> 1 ./(x*1000)), lera = l([r for r in 1:100])
# lera = 0.01
# function train_model!(x, y, NN, lera; x_test=x, y_test=y, batchsize=500, nepochs=10, update_step=10, beta=(0.9,0.999), lambda=0., loss_fct=MSELoss(), messages=true)
#     it = time()
#     Trainloss = [[],[]]
#     Testloss = [[],[]]
#     trainstate=nothing
#     for l in eachindex(lera)
#         messages && println("\n starting Loop ", l, ", learning rate: ", lera[l])
#         trainstate, trainloss, testloss = train_model!(x,y, model,params,states, x_test=x_test, y_test=y_test,
#         loss_fct=loss_fct, lera=lera[l], beta=beta, lambda=lambda, batchsize=batchsize, nepochs=nepochs, update_step=update_step, early_stopping=false, messages=messages)
#         Trainloss[1] = vcat(Trainloss[1], trainloss[1] .+(l-1)*(nepochs))
#         Testloss[1] = vcat(Testloss[1], testloss[1] .+(l-1)*(nepochs))
#         Trainloss[2] = vcat(Trainloss[2], trainloss[2])
#         Testloss[2] = vcat(Testloss[2], testloss[2])
#     end
#     ft = time()
#     tft = ft-it
#     println("Time for complete training: ", tft, "s \n")
#     return model_structure(trainstate.model, trainstate.parameters, trainstate.states), Trainloss, Testloss
# end


# adds input & output layers as repreprocessing accoring to  used preprocessing parameters
# NN: trained NN, var_prep_pars&K_prep_pars: preprocessing parameters as returned by get_train_test_set
function reprocess_model(NN_unprep; var_prep_pars=nothing, K_prep_pars=nothing)
    m = NN_unprep.model
    p = NN_unprep.parameters
    s = NN_unprep.states

    # if variables have been preprocessed: add input layer to preprocess input automatically
    if var_prep_pars!==nothing
        m_in = Scale(size(var_prep_pars[1]))
        p_in = (weight=1 ./var_prep_pars[2], bias=(-var_prep_pars[1] ./var_prep_pars[2]))
        _, s_in = Lux.setup(def_rng, m_in)

        m = Chain(m_in, m)
        p = (layer_1=p_in, layer_2=p)
        s = (layer_1=s_in,layer_2=s)
    end

    # if kernels have been preprocessed: add output layer to repreprocess output automatically
    if K_prep_pars!==nothing
        m_out = Scale(size(K_prep_pars[1]))
        p_out = (weight=K_prep_pars[2], bias=K_prep_pars[1])
        _, s_out = Lux.setup(def_rng, m_out)

        m = Chain(m, m_out)
        p = (layer_1=p, layer_2=p_out)
        s = (layer_1=s, layer_2=s_out)
    end

    # returns NN-structure with model, parameters, states with extra layers
    return model_structure(m,p,s)
end


# import export
function save_model(NN, filename)
    @save filename NN
end

function load_model(filename)
    @load filename NN
    return NN
end



####################### Kernelwise training #####################################


function NN_kernelwise(var_train_set, K_train_set, nb_hl, hl_dim; lera=0.001, beta1=0.9, beta2=0.999, lambda=0., loss_fct=MSELoss(), act_fct=leakyrelu, batchsize=500, nepochs=100, update_step=10, x_test=var_train_set, y_test=K_train_set, adapt_lera=false,  lera_trend=0.999, lera_update_step=10, early_stopping=true, messages=true)
    ti = time()
    var_dim = size(var_train_set)[1]
    K_dim = size(K_train_set)[1]

    nb_hl isa Number ? nb_hl = [nb_hl for i in 1:K_dim] : nb_hl=nb_hl
    hl_dim isa Number ? hl_dim = [hl_dim for i in 1:K_dim] : hl_dim=hl_dim
    lera isa Number ? lera = [lera for i in 1:K_dim] : lera=lera
    beta1 isa Number ? beta1 = [beta1 for i in 1:K_dim] : beta1=beta1
    beta2 isa Number ? beta2 = [beta2 for i in 1:K_dim] : beta2=beta2
    lambda isa Number ? lambda = [lambda for i in 1:K_dim] : lambda=lambda
    #epsilon isa Number ? epsilon = [epsilon for i in 1:K_dim] : epsilon=epsilon

    TRS = []
    TRL = []
    TEL = []


    for i in 1:K_dim
        messages && println("\n Kernel ", i, ":")
        m,p,s = initiate_model(var_dim, 1, nb_hl=nb_hl[i], hl_dim=hl_dim[i], act_fct=act_fct)
        trs,trl,tel = train_model!(var_train_set, reshape(K_train_set[i,:],1,:), model_structure(m,p,s), x_test=x_test, y_test=reshape(y_test[i,:],1,:),
        loss_fct=loss_fct, lera=lera[i], beta=(beta1[i],beta2[i]),lambda=lambda[i], batchsize=batchsize, nepochs=nepochs, update_step=update_step,  adapt_lera=adapt_lera, lera_trend=lera_trend, lera_update_step=lera_update_step, early_stopping=early_stopping, messages=messages)
        push!(TRS, trs)
        push!(TRL, trl)
        push!(TEL, tel)
    end


    tf = time()
    println("Time for complete training: ", tf-ti, "s \n")

    return TRS, TRL, TEL
end





####################### Hyperparameteroptimization #####################################

# THIS FUNCTION ONLY RANDOMLY TESTS GIVEN HYPERPARAMETERS !!!!
# IF YOU FIND A GOOD OPTIMIZATION PACKAGE IN JULA USE IT !!!!
# try the objective function trial() num_trials times with parameters taken form scen=options
# scen needs to be Dictionary with hyperparameters & options
# hyperparameters need to be correctly unpacked in trial() as entered in scen
# excepts: remove impossible combinations. form: [p->p[:h1]==a && p[:h2]==b] will remove all combinations where scen[:h1]=a and scen[:h2]=b
function trials(scens; excepts=[], num_trials=100)
    
    t0 = time()

    # # real randomizer (shuffled), but does not work on big options. Also, would need to be adjusted for the new dict structure instead of vectors
    # paropts = reshape([i for i in Iterators.product(values(scen)...)],:) 
    # for e in exepts
    #     filter!(!(e), paropts)
    # end
    # println("\n Number of options: ", length(paropts))
    # length(paropts) < num_trials && (num_trials = length(paropts))
    # opts_to_try = paropts[shuffle(eachindex(paropts))[1:num_trials]]

    #choose random option for each hyperparameter (only once), remove exeptions
    opts_to_try = unique([OrderedDict{Symbol,Any}(k=>v[rand(eachindex(v))] for (k,v) in scens) for i in 1:num_trials])
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


# function that runs & saves ALL given trials
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






################ TESTING ##########################################################################################################
function Z_matrix(XYZ)
    x_vals = sort(unique(v[1] for v in XYZ))
    y_vals = sort(unique(v[2] for v in XYZ))
    nx, ny = length(x_vals), length(y_vals)

    x_idx = Dict(p => i for (i,p) in enumerate(x_vals))
    y_idx = Dict(p => i for (i,p) in enumerate(y_vals))

    Z = fill(NaN, ny, nx)

    for (x,y,z) in XYZ
        Z[y_idx[y], x_idx[x]] = z
    end

    return x_vals, y_vals, Z
end


function plot_losses(trainloss, testloss)
    p = plot(trainloss[1], trainloss[2], yaxis=:log, label="Trainloss", xlabel="Epoch", ylabel="Loss")
    plot!(p, testloss[1], testloss[2], yaxis=:log, label="Testloss")
    return p
end



# usable only on sorted sets, plots one kernel in 2d
function plot_sorted_kernel_2d(var_set, K_set, i; labels = ["pT", "uʳ"], title="")
    vars = get_vars(var_set)
    return plot(vars[2],vars[1], reshape(K_set, (size(K_set)[1],length(vars[1]),length(vars[2])))[i,:,:], title=title, xlabel=labels[2], ylabel=labels[1], st=:surface, legend=:none)
end 

# usable on every set, takes long on big sets
function plot_kernel_2d(var_set, K_set, i; labels = ["pT", "uʳ"], title="")
    return plot(var_set[2,:], var_set[1,:], K_set[i,:], title=title, xlabel=labels[2], ylabel=labels[1], st=:scatter, legend=:none)
end


function plot_sorted_kernels_2d(var_set, K_set, K_labels; labels = ["pT", "uʳ"], title="")
    p = [plot_sorted_kernel_2d(var_set, K_set, i, title=K_labels[i], labels=labels) for i in axes(K_set,1)]
    return plot(p..., layout=(2,4), size=(2000,1000), plot_title=title)
end




function plot_sorted_kernels_ptur(var_set, K_set, K_labels; Tc_ind=1, Tk_ind=1, title="")
    vars = get_vars(var_set)
    ptur_dim = length(vars[1]) * length(vars[2])

    Tc_dim = length(vars) >=3 ? length(vars[3]) : 1
    temp_ind = ptur_dim*(Tc_ind-1) + ptur_dim*Tc_dim*(Tk_ind-1)

    if Tc_ind == 0 || Tc_ind > Tc_dim
        @error "Tchem index out of range"
    end

    length(vars) >=3 && println("Tchem: ", var_set[3,temp_ind+1], "GeV, Tkin: ", var_set[4,temp_ind+1], "GeV")

    plot_var_set = var_set[1:2,1+temp_ind:ptur_dim+temp_ind]
    plot_K_set = K_set[:,1+temp_ind:ptur_dim+temp_ind]

    return plot_sorted_kernels_2d(plot_var_set, plot_K_set, K_labels, title=title)
end


function plot_sorted_kernels_temps(var_set, K_set, K_labels; pt_ind=1, ur_ind=1, title="")
    vars = get_vars(var_set)
    var_dim = size(var_set)[2]
    pt_dim = length(vars[1])
    ptur_dim = length(vars[1]) * length(vars[2])

    if pt_ind == 0 || pt_ind > pt_dim
        @error "pt index out of range"
    end

    if ur_ind > length(vars[2])
        @error "ur index out of range"
    end

    ptur_ind_gen = (pt_ind+(ur_ind-1)*pt_dim ) % ptur_dim
    ptur_ind = [i for i in 1:var_dim if i%ptur_dim==ptur_ind_gen]

    println("pT: ", var_set[1,ptur_ind[1]], "GeV, ur: ", var_set[2,ptur_ind[1]], "GeV")

    plot_var_set = hcat([var_set[3:4,i] for i in ptur_ind]...)
    plot_K_set = hcat([K_set[:,i] for i in ptur_ind]...)

    return plot_sorted_kernels_2d(plot_var_set, plot_K_set, K_labels, labels=["Tchem", "Tkin"], title=title)
end



# compare all kernel predictions #vs=[var_set, vars_test_set], Ks = [K_set, K_test_set]
function compare_two_kernels_2d(vs, Ks, i; validation=false, loss=MSELoss(), labels=nothing, var_labels=["pT", "uʳ"])
    nb_plots = length(Ks)
    if labels===nothing
        labels = ["Interpolation", "Model prediction"] 
    end 

    plots = []
    for j in 1:nb_plots
        if validation
            p = plot_sorted_kernel_2d(vs[j], Ks[j], i, labels=var_labels)
        else
            p = plot_kernel_2d(vs[j], Ks[j], i, labels=var_labels)
        end
        plot!(p, title=labels[j])
        push!(plots, p)
    end

    if vs[1]==vs[2]
        println("Loss in Kernel ", i, " : ", loss(Ks[2][i], Ks[1][i]))
    end

    return plot(plots..., layout=(nb_plots,1), size=(500,nb_plots*500))
end



function show_kernels_pt(pt_range, K_func, trainstate; ur=0., plotlog=false)
    pt = sort(rand(def_rng, 200)*(pt_range[end]-pt_range[1]) .+ pt_range[1])
    ur = ones64(length(pt))*ur
    val_set = transpose(hcat(pt, ur))
    K_val_set = Kernels(val_set,K_func)
    K_val_NN, _ = Lux.apply(trainstate.model, val_set, trainstate.parameters, trainstate.states)
    
    for i in 1:8
        K_pt = K_val_set[i,:]
        K_NN_pt = K_val_NN[i,:]
        p = plot(pt,K_pt, label="Interpolation", xlabel="pT")
        plot!(pt,K_NN_pt, label="NN")
        if plotlog
            plot!(yaxis=:log, ylims=(1e-10,1e5))
        end
        display(p)
        sleep(3)
    end
end


function show_kernels_ur(ur_range, K_func, trainstate; pt=0., plotlog=false)
    ur = sort(rand(def_rng, 200)*(ur_range[end]-ur_range[1]) .+ ur_range[1])
    pt = ones64(length(ur))*pt
    val_set = transpose(hcat(pt, ur))
    K_val_set = Kernels(val_set,K_func)
    K_val_NN, _ = Lux.apply(trainstate.model, val_set, trainstate.parameters, trainstate.states)
    
    for i in 1:8
        K_ur = K_val_set[i,:]
        K_NN_ur = K_val_NN[i,:]
        p = plot(ur,K_ur, label="Interpolation", xlabel="uʳ")
        plot!(ur,K_NN_ur, label="NN")
        if plotlog
            plot!(yaxis=:log, ylims=(1e-10,1e5))
        end
        display(p)
        sleep(3)
    end
end




function show_sorted_kernels_pt(var_val_set, K_val_set, K_val_NN; ur_index=1, plotlog=false)
    n = axes(unique(var_val_set[1,:]),1)
    ur=ur_index-1
    pt = var_val_set[1,1+ur*n:n+ur*n]
    for i in 1:8
        K_pt = K_val_set[i,1+ur*n:n+ur*n]
        K_NN_pt = K_val_NN[i,1+ur*n:n+ur*n]
        p = plot(pt,K_pt, label="Interpolation", xlabel="pT")
        plot!(pt,K_NN_pt, label="NN")
        if plotlog
            plot!(yaxis=:log, ylims=(1e-10,1e5))
        end
        display(p)
        sleep(3)
    end
end


################### PCE plots #######################




function compare_kernels_PCE_ptur(var_set, K_func, trainstate, Tc, Tk; n=20)
    var_val_set_ptur = get_sorted_var_set(get_range(var_set[1:2,:]), n=n)
    var_val_set_tc = transpose([Tc for i in 1:n*n])
    var_val_set_tk = transpose([Tk for i in 1:n*n])
    var_val_set = vcat(var_val_set_ptur, var_val_set_tc, var_val_set_tk)
    K_val_set = Kernels(var_val_set, K_func)
    K_val_NN, _ = Lux.apply(trainstate.model, var_val_set, trainstate.parameters, trainstate.states)

    p1 = plot_sorted_kernels_ptur(var_val_set, K_val_set, K_labels, title="true kernels");
    p2 = plot_sorted_kernels_ptur(var_val_set, K_val_NN, K_labels, title="NN kernels");
    p3 = plot_sorted_kernels_ptur(var_val_set, K_val_NN ./K_val_set, K_labels, title="NN / true");
    return plot(p1,p2,p3, layout=(3,1), size=(2000,3000))
end


function compare_kernels_PCE_temps(var_set, K_func, trainstate, pt, ur; n=20)
    var_val_set_temps = get_sorted_var_set(get_range(var_set[3:4,:]), n=n)
    var_val_set_pt = transpose([pt for i in 1:n*n])
    var_val_set_ur = transpose([ur for i in 1:n*n])
    var_val_set = vcat( var_val_set_pt, var_val_set_ur, var_val_set_temps)
    K_val_set = Kernels(var_val_set, K_func)
    K_val_NN, _ = Lux.apply(trainstate.model, var_val_set, trainstate.parameters, trainstate.states)

    p1 = plot_sorted_kernels_temps(var_val_set, K_val_set, K_labels, title="true kernels");
    p2 = plot_sorted_kernels_temps(var_val_set, K_val_NN, K_labels, title="NN kernels");
    p3 = plot_sorted_kernels_temps(var_val_set, K_val_NN ./K_val_set, K_labels, title="NN / true");
    return plot(p1,p2,p3, layout=(3,1), size=(2000,3000))
end


