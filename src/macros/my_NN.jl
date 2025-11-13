include("/home/lisa/MA/NeuralNetwork/my_functions.jl");
include("/home/lisa/MA/Fluidum/my_main.jl");


data = readdlm("/home/lisa/MA/Data/Full_PCE/Kernels/pion_thermal_BG.txt", Float64);

var_set = data[1:4,:];
K_set = data[5:end,:];

K_func = extrapolate_interpolate_kernels(var_set, K_set);


#####################################################################################################################################################


var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars = get_train_test_set(var_set, K_set,
preprocess_vars=get_mean_std(var_set), preprocess_K=false, n_train=10000, n_test=10000);


my_NN = initiate_model(4, 8, nb_hl=6, hl_dim=256, act_fct=sigmoid_fast, 
hl_weight=initializer_gain(glorot_uniform, sigmoid_fast), hl_bias=randn32);



my_NN, trainloss, testloss = train_model!(var_train_set, K_train_set, my_NN, lera=0.00129155, beta=(0.99,0.9999),
batchsize=100, nepochs=10, early_stopping=true, x_test=var_test_set, y_test=K_test_set, loss_fct=MSELoss());


pl = plot_losses(trainloss, testloss)
#savefig(pl, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_learning_curve.png"))


Trainstate = reprocess_model(my_NN, var_prep_pars=var_prep_pars,K_prep_pars=K_prep_pars);
#save_model(Trainstate, "/home/lisa/MA/NeuralNetwork/pion_4D_BG/NN.jld2")




################


ps = compare_spectra_PCE(0.12,0.14, dic.pion, K_func, Trainstate; decays=false, pt_min=0.016, pt_max=3.727, plotlog=true)
ps = compare_spectra_PCE(0.12,0.14, dic.pion, K_func, Trainstate; decays=false, pt_min=0.016, pt_max=0.2, steps=200, comp_ratio=false, plotlog=false)
#savefig(ps, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_spectra.png"))

pk = compare_kernels_PCE_ptur(var_set, K_func, my_NN, 0.143, 0.125; n=20)
pk = compare_kernels_PCE_temps(var_set, K_func, Trainstate, 1.5, 1.5; n=20)
#savefig(pk, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_kernels.png"))



#save_model(Trainstate, String("/home/lisa/MA/NeuralNetwork/pion_4D_BG/" * whichtry * "_model.jld2"))


#########################################################################################################

# training loop ...
using Lux, Random, Optimisers, Statistics

# -------------------------
# 1. Load your data (x_raw, y_raw)
# -------------------------
# Suppose x_raw is size (4, N), y_raw is size (8, N)
# x_raw and y_raw must be Float32/64 arrays
# Example placeholders:
# x_raw = randn(Float32, 4, 1_000_000)
# y_raw = randn(Float32, 8, 1_000_000)
x_raw = var_set
y_raw = K_set

# -------------------------
# 2. Normalise inputs and outputs
# -------------------------
x_mean = mean(x_raw; dims=2)
x_std  = std(x_raw; dims=2) .+ eps()    # avoid /0
x_scaled = (x_raw .- x_mean) ./ x_std

# scale each output separately so small ones aren’t drowned out
y_mean = mean(y_raw; dims=2)
y_std  = std(y_raw; dims=2) .+ eps()
y_scaled = (y_raw .- y_mean) ./ y_std

# optional: per-output weights (after scaling, normally ones)
weights = ones(Float32, 8)


# -------------------------
# 3. Define model
# -------------------------
model = Chain(
    Dense(4, 256, tanh),
    Dense(256, 128, tanh),
    Dense(128, 64, tanh),
    Dense(64, 32, tanh),
    Dense(32, 8)  # linear output
)

rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# -------------------------
# 4. Loss function
# -------------------------
function loss_and_state(ps, st, x, y)
    ŷ, st_ = model(x, ps, st)
    l = mean(sum(weights .* (ŷ .- y).^2, dims=1))
    return l, st_
end

# -------------------------
# 5. Optimiser
# -------------------------
opt = Optimisers.Adam(1e-3)
opt_state = Optimisers.setup(opt, ps)

# -------------------------
# 6. Mini-batch iterator
# -------------------------
function minibatches(x, y; batchsize=1024)
    N = size(x, 2)
    idx = shuffle(1:N)
    return [(x[:, idx[i:min(i+batchsize-1, N)]],
             y[:, idx[i:min(i+batchsize-1, N)]]) 
            for i in 1:batchsize:N]
end

# -------------------------
# 7. Training loop
# -------------------------
nepochs = 20

for epoch in 1:nepochs
    total_loss = 0.0
    nbatches = 0
    for (x, y) in minibatches(x_scaled, y_scaled; batchsize=4096)

        # we need the current st in a Ref so gradient can capture it
        local_st = st
        grad_ps, back_loss = Zygote.gradient(ps) do p
            l, st_ = loss_and_state(p, local_st, x, y)
            # store new state externally (we’ll set st after)
            global new_st = st_
            return l
        end

        # update parameters
        ps, opt_state = Optimisers.update(opt, ps, grad_ps, opt_state)
        # update state to the one from forward pass
        st = new_st

        total_loss += back_loss  # back_loss == last returned l
        nbatches += 1
    end
    @info "Epoch $epoch: mean loss = $(total_loss/nbatches)"
end

# -------------------------
# 8. Prediction (auto-unscale)
# -------------------------
function predict(x_new)
    x_s = (x_new .- x_mean) ./ x_std
    ŷ_s, _ = model(x_s, ps, st)
    return ŷ_s .* y_std .+ y_mean
end

# Usage:
# y_pred = predict(x_new)  # x_new of size (4, M)

