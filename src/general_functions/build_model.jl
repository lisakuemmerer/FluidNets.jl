

##############################################################################################################
# # for reference: structure for model as included in Fluidum

# struct model_structure{}
#     model::Any
#     parameters::Any
#     states::Any
# end

# # can be applied equiv. to Trainstate -> include both options in structure
# model_structures = Union{Lux.Training.TrainState, model_structure}

# # easy apply: NN(x) gives prediction
# function (NN::model_structures)(x)
#     pred,_ = Lux.apply(NN.model, x, NN.parameters, NN.states)
#     return pred
# end

# # import export
# function save_model(NN, filename)
#     @save filename NN
# end
# function load_model(filename)
#     @load filename NN
#     return NN
# end





##############################################################################################################
# custom Loss functions


# logarithmic loss
MyLogLoss = GenericLossFunction((ŷ, y) -> (log(abs(ŷ - y)+1)));

# d^3 loss
MyM3ELoss = GenericLossFunction((ŷ, y) -> abs.(ŷ - y)^3);

# exponential loss
MyExpLoss = GenericLossFunction((ŷ, y) -> exp(abs(ŷ - y))-1);

# Cross-Entropy loss
# use only with out_act_fct=sigmoid or other scaled output
MyCELoss = GenericLossFunction((ŷ, y) -> -y*log(ŷ) - (1-y)*log(1-ŷ)); 

# weight on small kernalvalues for unprocessed kernels (or without shifted 0) 
# epsilon changes the weight, recommended value: 0.1 (for zeroabsmax normaliz.)
MyYweightLoss(epsilon) = GenericLossFunction((ŷ, y) -> abs2(ŷ - y) / (abs(y) + epsilon)); 
MyYweightLoss() = MyYweightLoss(1e-1)


# weight on high pT (variable 1), can be changed to put weight on other parameters as well
# w changes the weight. recommended value: 1 ( doubles high pT for minwidth normaliz. ))
# weight on high&low pT when used on with midhalfwidth normaliz.
function MyXweightLoss(m, p, s, (x, y_true), w)
    y_pred, st = m(x, p, s)
    weight = w*abs.(x[1,:]) .+1
    weight = reshape(weight,1,:)
    loss = mean(weight .* (y_pred .- y_true).^2)
    return loss, st, (;)
end
MyXweightLoss(w) = (m, p, s, (x, y_true)) -> MyXweightLoss(m, p, s, (x, y_true), w)
MyXweightLoss() = MyXweightLoss(1)





##############################################################################################################
# custom activation functions

# custom leakyrelu function with differing slope, = relu for a=0
leakyrelu_grad(a) = (b->leakyrelu(b,a))
ReLuTypes = Union{typeof(relu), typeof(leakyrelu), FluidNets.var"#leakyrelu_grad##0#leakyrelu_grad##1"{Float64}}





##############################################################################################################
# initializer

# calculates the gain corresponding to the activation function according to https://docs.pytorch.org/docs/stable/nn.init.html
# other activation functions can of course be included
ActfctswithgainTypes = Union{ReLuTypes, typeof(identity), typeof(sigmoid_fast), typeof(tanh_fast)}
function _gain(act_fct::ActfctswithgainTypes)
    act_fct isa ReLuTypes && (gain = sqrt(2/(1+act_fct(-1)^2)))
    act_fct == identity && (gain = 1)
    act_fct == sigmoid_fast && (gain = 1)
    act_fct == tanh_fast && (gain = 5/3)
    return gain
end


# gives the initializer with the correct gain depending on the activation function
# other initializers can be included of course
InitswithgainTypes = Union{typeof(kaiming_uniform), typeof(kaiming_normal), typeof(glorot_uniform), typeof(glorot_normal)}
function _initializer_gain(initializer::InitswithgainTypes, act_fct::ActfctswithgainTypes)
    init_with_gain(rng, dims...) = initializer(rng, dims...; gain=_gain(act_fct))
    return init_with_gain
end 





##############################################################################################################
# custom learning rate schedule functions

# linear descend starting from "start" with slope constant "red"/ reduce to "fin" after "pts"
linear(start, red) = x -> start - red*(x-1)
linear(start, fin, pts) =  x -> start - (start-fin)/(pts-1)*(x-1)

# exponential decay starting at "start"  with reduction = remaining percentage 
# (red=0.99 will reduce to 99% every step. for reduction to 99% every 10 steps put red = 0.99^(1/10) )
exponential(start, red) =  x -> start*red^(x-1)

# logarithmic decay starting from "start" reducing to "fin" after "pts"
logarithmic(start, fin, pts) =  x -> log10(pts/x)/log10(pts) * (start-fin) + fin

# cosine starting at "start" reducing to "fin" after "pts"
# should be used wiith repeat, otherwise schedule will increase learning rate after "pts"
cosine(start, fin, pts) =  x -> cos((x-1)/(pts-1)*pi) * (start-fin)/2 + (start+fin)/2
# should be used wiith repeat, otherwise schedule will increase learning rate after "pts"

# function that allows to stop function from propagating
# stop_on_value = false: for x>stop the value f(stop) will be repeated. useful to stop cosine after "pts"
# stop_on_value = true: returns "stop" instead of f(x) if f(x) is smaller. useful to keep linear with reduction from decreasing below certain value
function stop_function(f, x, stop, stop_on_value)
    if stop_on_value
        y = f(x)>=stop ? f(x) : stop
    else
        y = x<=stop ? f(x) : f(stop)
    end
    return y
end
stopped_function(f, stop; stop_on_value=false) = x-> stop_function(f, x, stop, stop_on_value)

#cosine that repeats last value after "pts"
stopped_cosine(start, fin, pts) = stopped_function(cosine(start, fin, pts), pts)





##############################################################################################################
# build & train & reprocess & save model


# initiate model for inp_dim=number of variables (input), out_dim=number of kernels (output)
# possible arguments: nb_hl=number of hidden layers, hl_dim=number of neurons in each hidden layer, 
# act_fct=activation function for hidden layers, out_act_fct=activation function for output layer, 
# param64=convert parameters to Float64, 
# hl_weight, hl_bias, inp_weight, inp_bias, out_weight, out_bias=initialization of weights and biases in respective layers. for hidden layers the function includes automatic computation of gain
function initiate_model(inp_dim, out_dim; nb_hl=5, hl_dim=32, act_fct=relu, in_act_fct=identity, out_act_fct=identity, param64=true, dropout=false, dropout_prob = 0.1,
                    hl_weight=nothing, hl_bias=nothing, inp_weight=nothing, inp_bias=nothing, out_weight=nothing, out_bias=nothing)

    # make sure weigts are initialized according to activation function
    hl_weight isa InitswithgainTypes && act_fct isa ActfctswithgainTypes && (hl_weight = _initializer_gain(hl_weight, act_fct))
    inp_weight isa InitswithgainTypes && in_act_fct isa ActfctswithgainTypes && (inp_weight = _initializer_gain(inp_weight, in_act_fct))
    out_weight isa InitswithgainTypes && out_act_fct isa ActfctswithgainTypes && (out_weight = _initializer_gain(out_weight, out_act_fct))

    # make chain of hidden layers and dropout layers (if dropout=true)
    hidden_layers = []
    for i in 1:nb_hl
        push!(hidden_layers, Dense(hl_dim=>hl_dim, act_fct, init_weight=hl_weight, init_bias=hl_bias))
        dropout==true && push!(hidden_layers, Dropout(dropout_prob))
    end
    # model: input layer, hidden layers, output layer
    model = Chain(Dense(inp_dim=>hl_dim, in_act_fct, init_weight=inp_weight, init_bias=inp_bias), 
    hidden_layers, Dense(hl_dim=>out_dim, out_act_fct, init_weight=out_weight, init_bias=out_bias))

    # initialize random parameters & empty states
    params, states = Lux.setup(Random.default_rng(),model)

    # change parameters to Float64 if param64 is true
    param64 && (params=Functors.fmap(x -> x isa AbstractArray ? Float64.(x) : x, params))

    # return model: input layer, hidden layers, output layers in form of Lux.Chain
    # return params: random initiazlied parameters in form of Lux.Parameters
    # return states: empty states in form of Lux.States
    return model_structure(model, params, states)
end


# train model on dataset x=variables, y=true Kernels, NN_init as initialized by initiate_model
# x_test, y_test = independent test set, 
# batchsize=number of datapoints in one batch, 
# nepochs=number of epochs to train,
# update_step=number of epochs after which test loss is calculated, 
# loss_fct=loss function to use, 
# lera=learning rate, beta,lambda,couple=AdamW parameters,
# adapt_lera: Int or false. if Int: after ... epochs learning rate will be adapted, according to lera_trend
# ---scalar: i.e. 0.999 for 0.1% decrease every #adapt_lera epoch
# ---vector: custom vector, i.e 10.^LinRange(-3,-5,1000) for logarithmic decrease. start should coincide with lera. the last value will be repeated for the remaining training if n_epochs > adatpt_lera*length(lera_vector)
# ---function: custom function depending on epoch i, i.e. cosine, log, exp...
# early_stopping: Int or false. if Int: will stop training if testloss > trainloss for ... instances
# patience: Int or false. if Int: stop training when testloss deos not decrease (less than 0.99) for ... instances
# messages = true if messages should be printed, optim_mode = true for more output values
function train_model!(x, y, NN_init; x_test=x, y_test=y, batchsize=500, nepochs=1000, 
    update_step=10, lera=0.001, beta=(0.9,0.999), lambda=0., couple=true,
    adapt_lera=false, lera_trend=0.999,
    loss_fct=MSELoss(), early_stopping=false, patience=false, messages=true, optim_mode=false)
    
    # build trainstate according to Lux training
    DatLoad = DataLoader((x,y), batchsize=batchsize, partial=false, shuffle=true) # from MLUtils; do not use on Slurm, f***s up everything
    train_state = Training.TrainState(NN_init.model, NN_init.parameters, NN_init.states, AdamW(lera,beta, lambda, couple=couple))
    
    # calculate initial loss
    in_loss,_,_ = loss_fct(train_state.model, train_state.parameters, train_state.states, (x, y))
    messages && println("\n initial loss ", in_loss)

    # initialize needed output arguments
    train_loss = ([],[])
    test_loss = ([],[])
    epoch_test_loss = 0.
    overfit_counter = 0
    converged_counter = 0
    stopping = false
    stop_requested = false
    t0 = time()


    for i in 1:nepochs
        # let run on background to enable ustom stop
        t = Threads.@spawn begin

            # train step: update on every batch in dataloader, batch loss is output 2
            # mean over whole dataset, append epoch & trainloss
            # use custom batching function if there are problems with MLUtils
            #epoch_train_loss = mean([Training.single_train_step!(AutoZygote(), loss_fct, (xbatch, ybatch), train_state)[2] for (xbatch, ybatch) in _batch_data(x,y,batchsize)])
            epoch_train_loss = mean([Training.single_train_step!(AutoZygote(), loss_fct, (xbatch, ybatch), train_state)[2] for (xbatch, ybatch) in DatLoad])
            push!(train_loss[2], epoch_train_loss)
            push!(train_loss[1], i)

            # test on testset
            if i%update_step == 0
                epoch_test_loss,_,_ = loss_fct(train_state.model, train_state.parameters, train_state.states, (x_test, y_test))
                push!(test_loss[2], epoch_test_loss)
                push!(test_loss[1], i+1)
                messages && println("after ", i, " epochs: time $(_format_seconds(time()-t0)) s, trainloss: ", epoch_train_loss, ", testloss (updated): ", epoch_test_loss)

                # test for fuck up - happens sometimes for bad hyperparameter choices
                if isnan(epoch_test_loss)
                    stopping = "broken"
                    return # Exit task
                end

                # compare loss to last test to check for convergence
                if length(test_loss[2]) >= 2
                    if test_loss[2][end]/test_loss[2][end-1] > 0.99
                        converged_counter += 1
                    else
                        converged_counter = 0
                    end
                end
                # if patience is true and counter reached break
                if patience!=false && converged_counter >= patience
                    stopping = "converged"
                    return # Exit task
                end

            end

            
            # compare testloss/trainloss for early stopping
            if i%update_step == 1
                if epoch_test_loss/epoch_train_loss > 1.
                    overfit_counter += 1
                else
                    overfit_counter = 0
                end
            end
            # stop if testloss > trainloss for 10 successive epochs
            if early_stopping!=false && overfit_counter >= early_stopping
                stopping = "overfit"
                return # Exit task
            end


            # update learning rate if adapt_lera is true
            #  && train_state.optimizer.eta >= 1e-5
            if adapt_lera!=false && i%adapt_lera==0 && i!=nepochs
                if lera_trend isa Number
                    eta = lera_trend*train_state.optimizer.eta
                    messages && println("updating learning rate")
                    Optimisers.adjust!(train_state, eta)
                    Lux.@set! train_state.optimizer = AdamW(eta, beta, lambda, couple=couple)
                elseif lera_trend isa Function
                    eta = lera_trend(i)
                    messages && println("updating learning rate")
                    Optimisers.adjust!(train_state, eta)
                    Lux.@set! train_state.optimizer = AdamW(eta, beta, lambda, couple=couple)
                elseif lera_trend isa Vector && i / lera_update_step + 1 <= length(lera_vector)
                    eta = lera_vector[Int(i / lera_update_step) + 1 ]
                    messages && println("updating learning rate")
                    Optimisers.adjust!(train_state, eta)
                    Lux.@set! train_state.optimizer = AdamW(eta, beta, lambda, couple=couple)
                end
            end
        end


        # redirect stop with "ctrl + c" to let it finish a started iteration and break training after loop
        try
            wait(t)
        catch e
            e isa InterruptException || rethrow()
            println("\nStop requested. Finishing current iteration...")
            stop_requested = true
            wait(t)
        end
        if stop_requested
            println("Stopping after iteration $i")
            break
        end
        if stopping !== false
            messages && println("Stopping loop (reason: $stopping)")
            break
        end
    end


    # measure time needed for training
    tft = time()-t0

    # calculate final loss
    out_loss,_,_ = loss_fct(train_state.model, train_state.parameters, train_state.states, (x, y))
    messages && println("Time for training:  $(_format_seconds(tft)) s\n initial loss: ", in_loss, " final loss: ", out_loss, " Improv: ", out_loss/in_loss, "\n")
    
    # return NN structure: updated model with trained parameters
    # return trainloss: tuple with (epoch, loss) for training loss (epoch & loss as array)
    # return testloss: tuple with (epoch, loss) for test loss (epoch & loss as array)
    # return time for training tft

    NN_trained = model_structure(train_state.model, train_state.parameters, train_state.states)
    optim_mode ? ret = (NN_trained, train_loss, test_loss, tft, stopping) : ret = (NN_trained, train_loss, test_loss)

    return ret
end

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
        _, s_in = Lux.setup(Random.default_rng(), m_in)

        m = Chain(m_in, m)
        p = (layer_1=p_in, layer_2=p)
        s = (layer_1=s_in,layer_2=s)
    end

    # if kernels have been preprocessed: add output layer to repreprocess output automatically
    if K_prep_pars!==nothing
        m_out = Scale(size(K_prep_pars[1]))
        p_out = (weight=K_prep_pars[2], bias=K_prep_pars[1])
        _, s_out = Lux.setup(Random.default_rng(), m_out)

        m = Chain(m, m_out)
        p = (layer_1=p, layer_2=p_out)
        s = (layer_1=s, layer_2=s_out)
    end

    # returns NN-structure with model, parameters, states with extra layers
    return model_structure(m,p,s)
end



# save and load loss arrays
function save_losses(trainloss, testloss, filename)
    open(filename, "w") do io
        writedlm(io, trainloss)
        writedlm(io, testloss)
    end
end
function load_losses(filename)
    t = readdlm(filename)
    trainloss = [Int.(t[1,:]), t[2,:]]
    testloss = [Int.(filter(t->t!="", t[3,:])), filter(t->t!="", t[4,:])]
    return trainloss, testloss
end



##############################################################################################################
# OTHER STUFF: kernelwise ( might not work anymore ? )


function NN_kernelwise(var_train_set, K_train_set, nb_hl, hl_dim; lera=0.001, beta1=0.9, beta2=0.999, lambda=0., loss_fct=MSELoss(), act_fct=leakyrelu, batchsize=500, nepochs=100, update_step=10, x_test=var_train_set, y_test=K_train_set, adapt_lera=100,  lera_trend=0.999,  early_stopping=100, patience=100, messages=true)
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
        loss_fct=loss_fct, lera=lera[i], beta=(beta1[i],beta2[i]),lambda=lambda[i], batchsize=batchsize, nepochs=nepochs, update_step=update_step,  adapt_lera=adapt_lera, lera_trend=lera_trend, early_stopping=early_stopping, patience=patience, messages=messages)
        push!(TRS, trs)
        push!(TRL, trl)
        push!(TEL, tel)
    end


    tf = time()
    println("Time for complete training: ", tf-ti, "s \n")

    return TRS, TRL, TEL
end

