

##############################################################################################################
# load data from files:
# var_set : variables in matrix form  (#vars, #samples)
# K_set : kernels in matrix form (#kernels, #samples)


# assuming you load from file as saved in get_files.jl:
# var_dim, K_dim: number of variables
function read_data(file, var_dim, K_dim)

    data = readdlm(file, Float64);

    if var_dim+K_dim != size(data,1)
        @warn "number of variables or kernels wrong?"
    end

    var_set = data[1:var_dim,:];
    K_set = data[var_dim+1:var_dim+K_dim,:];

    return var_set, K_set  
end






##############################################################################################################
# switch from variable arrays to grid (matrix) & back


# auxfuc to get unique variables in vector form
# returns array (number of variables) with arrays (number of unique values in variable)
function _get_vars(var_set)
    return [unique(var_set[i,:]) for i in axes(var_set,1)]
end

# auxfunc to return vars to matrix form with every combination of variables 
# result: matrix in order, first variable rises first
function _get_set(vars)
    reshape([r[i] for r in [i for i in Iterators.product(vars...)] for i in eachindex(vars)],length(vars),:)
end






##############################################################################################################
# get variable specs (min, max., mean, std..)


# auxfuc to get ranges of sets
# returns array (number of variables) of tuples (min, max) of each variable
function _get_range(set)
    return [(minimum(set[i,:]),maximum(set[i,:])) for i in axes(set,1)]
end


# auxfuc to get means&std of a set: 
# returns tuple of arrays: (mid-array, halfwidth-array)
# form as used for preprocessing to [-1,1]
function get_mid_halfwidth(set)
    ranges = _get_range(set)
    return ([(r[2]+r[1])/2 for r in ranges], [(r[2]-r[1])/2 for r in ranges])
end

# auxfuc to get minimum&widthof a set: 
# returns tuple of arrays: (mins-array, widths-array)
# form as used for preprocessing to [0,1]
function get_min_width(set)
    ranges = _get_range(set)
    return ([r[1] for r in ranges], [(r[2]-r[1]) for r in ranges])
end

# auxfuc to get zeros & absolute maximum of a set: 
# returns tuple of arrays: (zeros-array, absmax-array)
# form as used for preprocessing to maximal range in (-1,1) without shifting zero (low values)
function get_zero_absmax(set) 
    zero = [0.0 for i in 1:size(set, 1)]
    absmax = [maximum([abs(i) for i in g]) for g in _get_range(set)]
    return(zero, absmax)
end

## auxfuc to get means & std of a set: 
# returns tuple of arrays: (means-array, std-array)
# form as used for preprocessing to normalize
function get_mean_std(set)
    return (reshape(mean(set , dims=2),:), reshape(std(set , dims=2),:))
end






##############################################################################################################
# preprocessing & back


# normalize set x
# prep_pars: preprocessing parameters in form of ([m],[o]). will preprocess to x -> (x-m)/o
# returns preprocessed set, as well as parameters needed to reprocess the set
function preprocess(set; prep_pars=nothing)
    nb = size(set)[1]

    if prep_pars===nothing
        mins = [minimum(set[i,:]) for i in 1:nb]
        ranges = [maximum(set[i,:])-minimum(x[i,:]) for i in 1:nb]
        prep_pars = (mins, ranges)
    end

    prep = copy(set)
    for i in 1:nb
        prep[i,:] = (set[i,:].-prep_pars[1][i])/prep_pars[2][i]
    end

    return prep, prep_pars
end

# reprocess set x with given parameters prep_pars
function _reprocess(prep, prep_pars)
    x = copy(prep)
    for i in axes(prep,1)
        x[i,:] = prep[i,:]*prep_pars[2][i] .+ prep_pars[1][i]
    end
    return x
end






##############################################################################################################
# Interpolation (old style)


# compute array of extrapolated kernel functions
function extrapolate_interpolate_kernels(var_set, K_set)
    vars = _get_vars(var_set)
    K_interpol = [interpolate(Tuple(vars), reshape(K_set[i,:], Tuple([length(v) for v in vars])), Gridded(Linear()) ) for i in axes(K_set,1)]
    K_func = extrapolate.(K_interpol, Ref(Linear()))
    return K_func 
end


# calculate kernels for given variable set var_set & extrapolated kernels K_func
# returns kernels in matrix form (#kernels, #samples)
function Kernels(var_set, K_func)
    calc_K_set = zeros(length(K_func), size(var_set)[2])
    for i in axes(var_set,2)
        for j in eachindex(K_func)
            calc_K_set[j,i] = K_func[j](var_set[:,i]...)
        end
    end
    return calc_K_set
end






##############################################################################################################
# get random train & test set


# chose two random subsets from given dataset
# n_train/test: number of samples in train/test set. n_train+n_test must be smaller than number of samples
# preprocessing can be false (no prep) or a tuple with preprocessing parameters as given to preprocess()
# returns train & test sets for variables & kernels, as well as used preprocessing parameters
function get_train_test_set(var_set, K_set; preprocess_vars=false, preprocess_K=false, n_train=10000, n_test=10000)

    # get random unique indices for train/test set
    ind = shuffle(axes(var_set,2))[1:n_train+n_test]
    rand_vars = var_set[:,ind]
    rand_Ks = K_set[:,ind]

    # split into train & test set
    var_train_set = rand_vars[:,1:n_train]
    K_train_set = rand_Ks[:,1:n_train]
    var_test_set = rand_vars[:,n_train+1:n_train+n_test]
    K_test_set = rand_Ks[:,n_train+1:n_train+n_test]

    # preprocess sets if wanted
    if preprocess_vars == false
        var_prep_pars = nothing
    else 
        var_prep_pars = preprocess_vars
        var_train_set, _ = preprocess(var_train_set, prep_pars=var_prep_pars)
        var_test_set, _ = preprocess(var_test_set, prep_pars=var_prep_pars)
    end

    if preprocess_K == false
        K_prep_pars = nothing
    else
        K_prep_pars = preprocess_K
        K_train_set, _  = preprocess(K_train_set, prep_pars=K_prep_pars)
        K_test_set, _ = preprocess(K_test_set, prep_pars=K_prep_pars)
    end

    return var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars
end





##############################################################################################################
# batches


function _batch_data(x,y,batchsize)
    
    shuffled_inds = shuffle(axes(x,2))

    shuffled_x = x[:,shuffled_inds][:,1:size(x,2)-size(x,2)%batchsize]
    shuffled_y = y[:,shuffled_inds][:,1:size(y,2)-size(y,2)%batchsize]

    reshaped_shuffled_x = reshape(shuffled_x, size(x,1),batchsize,:)
    reshaped_shuffled_y = reshape(shuffled_y, size(y,1),batchsize,:)
    D = [(reshaped_shuffled_x[:,:,i], reshaped_shuffled_y[:,:,i]) for i in axes(reshaped_shuffled_x, 3)]

    return D
end






##############################################################################################################
# other stuff: get sets based on Interpolation ( might not work anymore ? )


# compute random variable set
# var_range can be array range as in _get_range, or variable arrays as in _get_vars
# n: number of samples, rng: random number generator
# returns random variable set in matrix form (#vars, #samples)
function compute_var_set(var_range; n=100)
    v_rand = [rand(n)*(v[end]-v[1]) .+ v[1] for v in var_range]
    return stack(v_rand, dims=1)
end


# compute sorted variable set (grid)
# var_range can be array range as in _get_range, or variable arrays as in _get_vars
# n: number of samples - each combination is computed. will return n^(#vars) samples
# returns random variable set in matrix form (#vars, #samples)
function compute_sorted_var_set(var_range; n=10)
    v_rand = [sort(unique(rand(n)*(v[end]-v[1]) .+ v[1])) for v in var_range]
    return _get_set(v_rand)
end



# random variables with corresponding kernels in mtrx form (#vars, n), (#kernels, n)
function get_val_set(var_range, K_func, trainstate::model_structures; n=100)
    var_val_set = compute_sorted_var_set(var_range, n=n)
    K_val_set = Kernels(var_val_set, K_func)
    K_val_NN,_ = Lux.apply(trainstate.model, var_val_set, trainstate.parameters, trainstate.states)
    return var_val_set, K_val_set, K_val_NN
end
function get_val_set(var_range, K_func, Trainstate::Vector{Any}; n=100)
    var_val_set = compute_sorted_var_set(var_range, n=n)
    K_val_set = Kernels(var_val_set, K_func)
    K_val_NN = vcat([Lux.apply(t.model, var_val_set, t.parameters, t.states)[1] for t in Trainstate]...)
    return var_val_set, K_val_set, K_val_NN # can get Testset with 'vcat(var_test_set, K_test_set)'
end


# compute two random sets from extrapolated kernels
# preprocess true/false
# prep_range: range to preprocess on, can be given as _get_range(set) for normal. on (0,1) for given set-range
# if no prep range is given, but preprocess=true, normalization on ranges of train/testset is used for kernels, var_range is used for variables
# K_scale: extra weight on kernels, #kernel-dim. array or scalar
# n_train, n_test: number of samples in train/test set
# returns train & test sets for variables & kernels, as well as used preprocessing parameters
function compute_train_test_set(var_range::Vector, K_func::Vector; sort=false, preprocess_vars=false, var_prep_range=nothing, preprocess_K=false, K_prep_range=nothing, K_scale=1, n_train=10000, n_test=10000)
    var_dim = size(var_range)[1]

    K_dim = size(K_func)[1]

    sort==true ? var_train_set = compute_sorted_var_set(var_range, n=n_train) : var_train_set = compute_var_set(var_range, n=n_train)
    K_train_set = Kernels(var_train_set, K_func)

    sort==true ? var_test_set = compute_sorted_var_set(var_range, n=n_test) : var_test_set = compute_var_set(var_range, n=n_test)
    K_test_set = Kernels(var_test_set, K_func)

    if preprocess_vars
        if var_prep_range===nothing
            var_prep_range=var_range
        end
        var_prep_pars = ( [var_prep_range[i][1] for i in 1:var_dim], [var_prep_range[i][end]-var_prep_range[i][1] for i in 1:var_dim] )
        var_train_set, _ = preprocess(var_train_set, prep_pars=var_prep_pars)
        var_test_set, _ = preprocess(var_test_set, prep_pars=var_prep_pars)
    else
        var_prep_pars = nothing
    end

    if preprocess_K
        if K_prep_range===nothing
            K_prep_range  = [(minimum(vcat(K_train_set[i,:], K_test_set[i,:])), maximum(vcat(K_train_set[i,:], K_test_set[i,:]))) for i in 1:K_dim]
        end
        if K_scale isa Number
            K_scale=ones64(K_dim)*K_scale
        end
        K_prep_pars = ( [K_prep_range[i][1] for i in 1:K_dim], [(K_prep_range[i][end]-K_prep_range[i][1])./K_scale[i] for i in 1:K_dim] )
        K_train_set, _  = preprocess(K_train_set, prep_pars=K_prep_pars)
        K_test_set, _ = preprocess(K_test_set, prep_pars=K_prep_pars)
    else
        K_prep_pars = nothing
    end

    return var_train_set, K_train_set, var_test_set, K_test_set, var_prep_pars, K_prep_pars
end




