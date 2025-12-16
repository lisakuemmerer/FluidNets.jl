

##############################################################################################################
# losses


function plot_losses(trainloss, testloss)
    p = plot(trainloss[1], trainloss[2], yaxis=:log, label="Trainloss", xlabel="Epoch", ylabel="Loss")
    plot!(p, testloss[1], testloss[2], yaxis=:log, label="Testloss")
    return p
end





##############################################################################################################
# plot BG kernels of 2D data


# plots one kernel in 2d, usable only on sorted (on a grid) sets, 
function _plot_sorted_kernel_2d(var_set, K_set, i; labels = ["pT", "uʳ"], title="", surface=true)
    vars = _get_vars(var_set)
    if surface
        p = plot(vars[2],vars[1], reshape(K_set, (size(K_set)[1],length(vars[1]),length(vars[2])))[i,:,:], title=title, xlabel=labels[2], ylabel=labels[1], st=:surface, legend=:none)
    else
        p = plot(vars[2],vars[1], reshape(K_set, (size(K_set)[1],length(vars[1]),length(vars[2])))[i,:,:], title=title, xlabel=labels[2], ylabel=labels[1], st=:heatmap)
    end
    return p
end 


# same, but usable on every set, takes long on big sets
function _plot_kernel_2d(var_set, K_set, i; labels = ["pT", "uʳ"], title="")
    return plot(var_set[2,:], var_set[1,:], K_set[i,:], title=title, xlabel=labels[2], ylabel=labels[1], st=:scatter, legend=:none)
end


# plots 8 kernels in 2d
function _plot_sorted_kernels_2d(var_set, K_set, K_labels; labels = ["pT", "uʳ"], title="", surface=true)
    p = [_plot_sorted_kernel_2d(var_set, K_set, i, title=K_labels[i], labels=labels, surface=surface) for i in axes(K_set,1)][1:8]
    if surface
        pl = plot(p..., layout=(2,4), size=(2000,1000), plot_title=title)
    else
        pl = plot(p..., layout=(2,4), size=(3000,1000), left_margin=20*Plots.mm, bottom_margin=20*Plots.mm, plot_title=title)
        #pl = plot(p...)
    end
    return pl
end





##############################################################################################################
# plot BG kernels of 4D data


#plots 8 kernels in 2d (pt,ur) given indices for Tchem & Tkin, prints according temperatures
function plot_sorted_kernels_ptur(var_set, K_set, K_labels; Tc_ind=1, Tk_ind=1, title="", printout=true)
    vars = _get_vars(var_set)
    ptur_dim = length(vars[1]) * length(vars[2])

    Tc_dim = length(vars) >=3 ? length(vars[3]) : 1
    temp_ind = ptur_dim*(Tc_ind-1) + ptur_dim*Tc_dim*(Tk_ind-1)

    if Tc_ind == 0 || Tc_ind > Tc_dim
        @error "Tchem index out of range"
    end

    length(vars) >=3 && printout==true && println("Tchem: ", var_set[3,temp_ind+1], "GeV, Tkin: ", var_set[4,temp_ind+1], "GeV")

    plot_var_set = var_set[1:2,1+temp_ind:ptur_dim+temp_ind]
    plot_K_set = K_set[:,1+temp_ind:ptur_dim+temp_ind]

    return _plot_sorted_kernels_2d(plot_var_set, plot_K_set, K_labels, title=title)
end


#plots 8 kernels in 2d (Tchem, Tkin) given indices for pt,ur, prints according pt,ur
function plot_sorted_kernels_temps(var_set, K_set, K_labels; pt_ind=1, ur_ind=1, title="", printout=true)
    vars = _get_vars(var_set)
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

    printout==true && println("pT: ", var_set[1,ptur_ind[1]], "GeV, ur: ", var_set[2,ptur_ind[1]], "GeV")

    plot_var_set = hcat([var_set[3:4,i] for i in ptur_ind]...)
    plot_K_set = hcat([K_set[:,i] for i in ptur_ind]...)

    return _plot_sorted_kernels_2d(plot_var_set, plot_K_set, K_labels, labels=["Tchem", "Tkin"], title=title)
end




##############################################################################################################
# comparison of prediction & interpolation of 4D data, BG kernels


# compare prediction and interpolation for 8 kernels in pt,ur given values for Tchem, Tkin
# returns 8k for interpol, pred & ratio
function compare_kernels_ptur(var_set, K_func, model, Tc, Tk; n=20, show_mse=false)
    var_val_set_ptur = compute_sorted_var_set(_get_range(var_set[1:2,:]), n=n)
    var_val_set_tc = transpose([Tc for i in 1:n*n])
    var_val_set_tk = transpose([Tk for i in 1:n*n])
    var_val_set = vcat(var_val_set_ptur, var_val_set_tc, var_val_set_tk)
    K_val_set = Kernels(var_val_set, K_func)
    K_val_NN = model(var_val_set)

    Losses = [MSELoss()(K_val_NN[i,:], K_val_set[i,:]) for i in axes(K_val_NN, 1)]
    println("Loss in each kernel for Tchem=$(Tc), Tkin=$(Tk):")
    for i in axes(K_val_NN,1)
        println(K_labels[i], ": ", Losses[i])
    end

    if show_mse
        p = _plot_sorted_kernels_2d(var_val_set_ptur, (K_val_NN .-K_val_set).^2, K_labels, title="MSE between prediction and interpolation", surface=false)
    else
        p1 = _plot_sorted_kernels_2d(var_val_set_ptur, K_val_set, K_labels, title="True kernels");
        p2 = _plot_sorted_kernels_2d(var_val_set_ptur, K_val_NN, K_labels, title="NN kernels");
        p3 = _plot_sorted_kernels_2d(var_val_set_ptur, K_val_NN ./K_val_set, K_labels, title="NN / true");
        p = plot(p1,p2,p3, layout=(3,1), size=(2000,3000))
    end

    return p
end


# compare prediction and interpolation for 8 kernels in Tchem, Tkin given values for pt, ur
# returns 8k for interpol, pred & ratio
function compare_kernels_temps(var_set, K_func, model, pt, ur; n=20, show_mse=false)
    var_val_set_temps = compute_sorted_var_set(_get_range(var_set[3:4,:]), n=n)
    var_val_set_pt = transpose([pt for i in 1:n*n])
    var_val_set_ur = transpose([ur for i in 1:n*n])
    var_val_set = vcat( var_val_set_pt, var_val_set_ur, var_val_set_temps)
    K_val_set = Kernels(var_val_set, K_func)
    K_val_NN = model(var_val_set)

    Losses = [MSELoss()(K_val_NN[i,:], K_val_set[i,:]) for i in axes(K_val_NN, 1)]
    println("Loss in each kernel for pt=$(pt), ur=$(ur):")
    for i in axes(K_val_NN,1)
        println(K_labels[i], ": ", Losses[i])
    end

    if show_mse
        p = _plot_sorted_kernels_2d(var_val_set_temps, (K_val_NN .-K_val_set).^2, K_labels, title="MSE between prediction and interpolation", surface=false)
    else
        p1 = _plot_sorted_kernels_2d(var_val_set_temps, K_val_set, K_labels, title="True kernels");
        p2 = _plot_sorted_kernels_2d(var_val_set_temps, K_val_NN, K_labels, title="NN kernels");
        p3 = _plot_sorted_kernels_2d(var_val_set_temps, K_val_NN ./K_val_set, K_labels, title="NN / true");
        p = plot(p1,p2,p3, layout=(3,1), size=(2000,3000))
    end

    return p
end





##############################################################################################################
# OTHER STUFF: compare kernels of 2 sets on 2d data ( might not work anymore ? )


# compares one kernel predictions #vs=[var_set, vars_test_set], Ks = [K_set, K_test_set]
function compare_two_kernels_2d(vs, Ks, i; validation=false, loss=MSELoss(), labels=nothing, var_labels=["pT", "uʳ"])
    nb_plots = length(Ks)
    if labels===nothing
        labels = ["Interpolation", "Model prediction"] 
    end 

    plots = []
    for j in 1:nb_plots
        if validation
            p = _plot_sorted_kernel_2d(vs[j], Ks[j], i, labels=var_labels)
        else
            p = _plot_kernel_2d(vs[j], Ks[j], i, labels=var_labels)
        end
        plot!(p, title=labels[j])
        push!(plots, p)
    end

    if vs[1]==vs[2]
        println("Loss in Kernel ", i, " : ", loss(Ks[2][i], Ks[1][i]))
    end

    return plot(plots..., layout=(nb_plots,1), size=(500,nb_plots*500))
end


# compare kernel prediction to interpolation in 1d (pt), given ur
# runs through plot for each kernel
function show_kernels_pt(pt_range, K_func, trainstate; ur=0., plotlog=false)
    pt = sort(rand(200)*(pt_range[end]-pt_range[1]) .+ pt_range[1])
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


# compare kernel prediction to interpolation in 1d (ur), given pt
# runs through plot for each kernel
function show_kernels_ur(ur_range, K_func, trainstate; pt=0., plotlog=false)
    ur = sort(rand(200)*(ur_range[end]-ur_range[1]) .+ ur_range[1])
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


# compare kernel prediction to interpolation in 1d (pt), given ur index
# used on precomputed sorted validation set (should be somewhere in process_data)
# runs through plot for each kernel
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




