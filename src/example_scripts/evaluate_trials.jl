using FluidNets

Trials = load_trials("/home/lisa/MA/Final/hyperopt/pion_total/trial_all.jld2");


# get hyppars & results
results = [:endloss, :improv, :tft, :overfit] ;
hyppars = [h for h in filter(!(k->k in results || k==:loss_fct), keys(first(Trials)))];
scen = get_options(Trials, hyppars)




##################################################################################################################
# evaluate
# :improv should be used as verific when different loss functions are compared

# best trial:
sortby(Trials)[1]

# best 5 Trials:
sT = sortby(Trials)[1:5]
for s in keys(sT[1])
    println(s, " ", [T[s] for T in sT])
end





#################################################################################################################
# split "broken" trials

# Trials_broken = filter(t->t[:overfit]=="broken", Trials)
# Trials = filter(!(t->t[:overfit]=="broken"), Trials)

Trials_broken = filter(t->isnan(t[:endloss]), Trials)
Trials = filter(!(t->isnan(t[:endloss])), Trials)


########

# plot occurance of parameter choices in the 'broken' Trials
plot_all_hyppars(hist_occurance, hyppars, Trials_broken)


# plot occurance for corraltion of two hyppars
hist_correlation_occurance(:nb_hl, :hl_dim, Trials_broken)
# # plot occurance in 2d correlation for all combinations -- loops over 144 plots, takes a looooong time -- 
# for h1 in hyppars
#     for h2 in filter(!(x->x==h1), hyppars)
#         p = hist_correlation_occurance(h1,h2,Trials_broken)
#         display(p)
#         sleep(3)
#     end
# end






#################################################################################################################
# evaluate working trials


# scatter loss for all choices (all / best 100)
plot_all_hyppars(plot_hyppar, hyppars, Trials, title="all hyppars")
plot_all_hyppars(plot_hyppar, hyppars, Trials, title="all hyppars", trialmax=100)

# percentage course in best trials (all / best 100)
plot_all_hyppars(plot_course_in_best_trials, hyppars, Trials, title="all hyppars")
plot_all_hyppars(plot_course_in_best_trials, hyppars, Trials, title="all hyppars", trialmax=100)


# histogramm loss for all choices
plot_all_hyppars(hist_hyppar, hyppars, Trials)


# histogramm occurence of choice in best 100 trials
plot_all_hyppars(hist_occurance, hyppars, Trials, trialmax=100)


# mean result for correlation
plot_correlation(:nb_hl, :hl_dim, Trials)
# # plot mean of loss for correlation of all combintaions -- loops over 144 plots, takes a looooong time -- 
# for h1 in hyppars
#     for h2 in filter(!(x->x==h1), hyppars)
#         p = plot_correlation(h1,h2,Trials)
#         display(p)
#         sleep(3)
#     end
# end


# correlation of occurance in best 100 trials
hist_correlation_occurance(:nb_hl, :hl_dim, Trials, trialmax=100)
# # histogramm correlating occurence in best 100 trials for all combintaions-- loops over 144 plots, takes a looooong time -- 
# for h1 in hyppars
#     for h2 in filter(!(x->x==h1), hyppars)
#         p = hist_correlation_occurance(h1,h2,Trials, trialmax=100)
#         display(p)
#         sleep(3)
#     end
# end


