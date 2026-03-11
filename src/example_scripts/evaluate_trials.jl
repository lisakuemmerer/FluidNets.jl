using FluidNets

Trials = load_trials("/home/lisa/MA/Final/hyperopt/pion_total/trial_all.jld2");


# get hyppars & results
results = [:endloss, :improv, :tft, :overfit];
hyppars = [h for h in filter(!(k->k in results || k==:loss_fct), keys(first(Trials)))];
scen = get_options(Trials, hyppars)




##################################################################################################################
# evaluate
# :improv should be used as verific when different loss functions are compared


############################
# best loss results:


# best trial:
sortby(Trials)[1]


# best 5 Trials:
sT = sortby(Trials)[1:5]
for s in keys(sT[1])
    println(s, " ", [T[s] for T in sT])
end



############################
# performance based on loss 


# scatter loss for all choices (all / best 100)
plot_all_hyppars(plot_hyppar, hyppars, Trials, plottitle="all hyppars")
plot_all_hyppars(plot_hyppar, hyppars, Trials, plottitle="all hyppars", trialmax=100)


# histogramm loss for all choices
plot_all_hyppars(hist_hyppar, hyppars, Trials)


# mean result for correlation - look at combis you think might be interesting
plot_correlation(:nb_hl, :hl_dim, Trials)
# loop over all combinations for one hyperparameter
loop_one_2D(plot_correlation, :act_fct, hyppars, Trials)
# loop over all combinations
loop_all_2D(plot_correlation, hyppars, Trials)





############################
# performance based on best trials


# percentage course in best trials (all / best 100)
plot_all_hyppars(plot_course_in_best_trials, hyppars, Trials, plottitle="all hyppars")
plot_all_hyppars(plot_course_in_best_trials, hyppars, Trials, plottitle="all hyppars", trialmax=100)


# histogramm occurence of choice in best 100 trials
plot_all_hyppars(hist_occurance, hyppars, Trials, trialmax=100)


# correlation of occurance in best 100 trials - look at combis you think might be interesting
hist_correlation_occurance(:nb_hl, :hl_dim, Trials, trialmax=100)
# loop over combis for pne hyperparameter
loop_one_2D(hist_correlation_occurance :act_fct, hyppars, Trials, trialmax=100)
# loop over all combinations
loop_all_2D(hist_correlation_occurance, hyppars, Trials, trialmax=100)



