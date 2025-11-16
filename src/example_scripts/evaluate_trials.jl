using FluidNets

Trials_all = load_trials("/home/lisa/MA/NeuralNetwork/hyperparameter/Trials.jld2")


hyppars = [k for k in keys(Trials_all[1])][1:13]
results = [k for k in keys(Trials_all[1])][14:end]

function get_options(Trials, hyppars)
    scen = OrderedDict{Symbol, Any}(h=>[] for h in hyppars)
    for h in hyppars
        for t in Trials_all
            t[h] in scen[h] ? nothing : push!(scen[h], t[h])
        end
        sort!(scen[h])
    end
    return scen
end
scen = get_options(Trials_all, hyppars)

# # remove double hyppar options
# unique_entries = unique(d -> Tuple(d[k] for k in hyppars), sort(Trials, by = x->x[:endloss]))
# # get even count of options
#Trials_all = Trials_all[shuffle([i for i in eachindex(Trials_all)])][5000]


best_endloss = argmin(x->x[:endloss], Trials_all)
best_improv = argmin(x->x[:improv], Trials_all)


#####################################################################################################################
#test percentage in first --- entries:

function percent(hyppar, Trials; sortby=:endloss, firsttrials=100)
    Trials_sorted = sort(Trials, by = x -> x[sortby])
    hypparchoices = [t[hyppar] for t in Trials_sorted]
    counts = countmap(hypparchoices)
    choices = keys(counts)
    counts_best = countmap(hypparchoices[1:firsttrials])
    for c in choices
        c in keys(counts_best) || (counts_best[c] = 0)
    end
    return OrderedDict{Any,Any}(c=>round(counts_best[c]/counts[c]*100, digits=2) for c in choices)
end

# plot course of percentage contained
function plot_percents(hyppar, Trials; sortby=:endloss, trialmax=1000)
    perc_scens = [n->percent(hyppar, Trials, sortby=sortby, firsttrials=n)[s] for s in scen[hyppar]]
    n = LinRange(0:10:trialmax)
    plot(n, [p.(n) for p in perc_scens], label=reduce(hcat, scen[hyppar]), title="Percentage in best trials ($(String(sortby)))",
        xlabel="number of best trials", ylabel="contained percentage")
end


percents = OrderedDict{Symbol,Any}(h=>percent(h, Trials_all) for h in hyppars)
best_percents = OrderedDict{Symbol,Any}(h=>maximum(percents[h])[1] for h in hyppars)


# part into lossfct, all others
hyppars_noloss = filter(!(x->x==:loss_fct), hyppars)


# plot all other best 100
plot([plot([t[h] for t in Trials_all][1:100], [t[:endloss] for t in Trials_all][1:100], yscale=:log10, st=:scatter, label="", title=String(h)) 
    for h in hyppars_noloss]..., layout=(4,3), size=(2000,1800), plot_title="Best 100 trials (endloss)")
#savefig("/home/lisa/MA/NeuralNetwork/hyperparameter/best_100_all.png")

#plot course for all
plot([plot_percents(h, Trials_all) for h in hyppars_noloss]..., title="", xlabel="", ylabel="", layout=(4,3), size=(2000,1800), 
    plot_title="Percentage in best trials (endloss)")
#savefig("/home/lisa/MA/NeuralNetwork/hyperparameter/percentage_course_all.png")



# for loss fct: do with improv
percent(:loss_fct, Trials_all, sortby=:improv)
plot_percents(:loss_fct, Trials_all, sortby=:improv)
#savefig("/home/lisa/MA/NeuralNetwork/hyperparameter/percentage_course_lossfct.png")

plot([plot([t[:loss_fct] for t in Trials_all[1:100]], [t[s] for t in Trials_all[1:100]], 
    st=:scatter, yscale=:log10, label="", ylabel=Symbol(s)) for s in [:endloss, :improv]]..., 
    layout=(2,1), size=(600,600), plot_title="Lossfct in best 100")
#savefig("/home/lisa/MA/NeuralNetwork/hyperparameter/best_100_lossfct.png")


########################
#decide for MSE

Trials_mse = sort(filter(x->x[:loss_fct]=="mse", Trials_all), by=x->x[:endloss])

percents_mse = OrderedDict{Symbol,Any}(h=>percent(h, Trials_mse) for h in hyppars)
best_percents_mse = OrderedDict{Symbol,Any}(h=>maximum(percents_mse[h])[1] for h in hyppars)





# plot all other best 100
plot([plot([t[h] for t in Trials_mse][1:100], [t[:endloss] for t in Trials_mse][1:100], yscale=:log10, st=:scatter, label="", title=String(h)) 
    for h in hyppars_noloss]..., layout=(4,3), size=(2000,1800), plot_title="Best 100 MSE trials (endloss)")
#savefig("/home/lisa/MA/NeuralNetwork/hyperparameter/best_100_mse.png")

#plot course for all
plot([plot_percents(h, Trials_mse) for h in hyppars_noloss]..., title="", xlabel="", ylabel="", layout=(4,3), size=(2000,1800), 
    plot_title="Percentage in best MSE trials (endloss)")
#savefig("/home/lisa/MA/NeuralNetwork/hyperparameter/percentage_course_mse.png")




##########################################################################################################
#test correlation

function plot_combination(v1,v2, Trials)
    var_comb_counted = countmap((t[v1], t[v2]) for t in Trials)
    var_combs = [k for k in keys(var_comb_counted)]

    res = []
    for k in var_combs
        var_comb_trials = filter(t->t[v1]==k[1]&&t[v2]==k[2], Trials)
        comb_result_summed = sum(t[:endloss] for t in var_comb_trials)/var_comb_counted[k]
        push!(res, (k[1], k[2],comb_result_summed))
    end

    x_vals, y_vals, Z = Z_matrix(res)
    Zlog = log10.(Z)
    nx,ny=length(x_vals), length(y_vals)
    p =  heatmap(1:nx, 1:ny, Zlog;
    xticks=(1:nx, string.(x_vals)),yticks=(1:ny, string.(y_vals)),
    xlabel=String(v1), ylabel=String(v2), colorbar_title="Endloss (logarithmic)",
    c=:viridis, aspect_ratio=:equal)
    return p
end


plot_combination(:prep_vars, :prep_K, Trials_all)
plot_combination(:nb_hl, :hl_dim, Trials_all)
plot(plot_combination(:initializer_weight, :initializer_bias, Trials_all), size=(1000,1000))
plot(plot_combination(:lera, :lambda, Trials_all), size=(1000,300))
plot_combination(:act_fct, :batchsize, Trials_all)
plot_combination(:beta1, :beta2, Trials_all)







##################################################################################################
# PCE.... deepsave ? 


#trial_all = load_trials("/home/lisa/MA/NeuralNetwork/pion_4D_BG/trial_all.json");

trial_all_sorted = sort(trial_all, by=t -> t[:testloss][2][end]/t[:testloss][2][1]);

plot_losses(trial_all_sorted[1][:trainloss], trial_all_sorted[1][:testloss])

trial_all_sorted[1]
   
##########################



plot([t[:hl_dim] for t in trialdatamodel], [t[:testloss][2][end]/t[:testloss][2][1] for t in trialdatamodel], 
yscale=:log10, st=:scatter, label="")
plot([t[:hl_dim] for t in trialdatamodel], [t[:testloss][2][end] for t in trialdatamodel], 
yscale=:log10, st=:scatter, label="")



trials_inorder_relative = sort(trialdatamodel, by=t -> t[:testloss][2][end]/t[:testloss][2][1])
trials_inorder_endtstls = sort(trialdatamodel, by=t -> t[:testloss][2][end])
trials_notover = [t for t in trialdatamodel if t[:overfit] == false]

function test(i; relative=false)
    relative ? t = trials_inorder_relative[i] : t = trials_inorder_endtstls[i]
    println("  Test loss: $(t[:testloss][2][end]), Improv: $(t[:testloss][2][end]/t[:testloss][2][1])")
    println("  Time for training: $(t[:tft]) seconds, Overfit: $(t[:overfit])")
    println("  Hyperparameters: $(t[:prep_vars]), $(t[:prep_K]), $(t[:nb_hl]), $(t[:hl_dim]), $(t[:act_fct])")
    return plot_losses(t[:trainloss], t[:testloss])
end


p = [test(i, relative=true) for i in 1:8]
plot(p..., layout=(4,2), size=(750,1000))
