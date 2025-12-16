using FluidNets

Trials = load_trials("/home/lisa/MA/Final/hyperopt/pion_thermal/trials_all.jld2");

# # remove double hyppar options - only needed the first time
# unique_entries = unique(d -> Tuple(d[k] for k in hyppars), sort(Trials, by = x->x[:endloss]))
# # get even count of options
# Trials = Trials[shuffle([i for i in eachindex(Trials)])][5000]

# get hyppars & results
results = [:endloss, :improv, :tft, :overfit] 
hyppars = [h for h in filter(!(k->k in results || k==:loss_fct), keys(first(Trials)))]
scen = get_options(Trials, hyppars)
 

#########################
# evaluate
# :improv should be used as verific when different loss functions are compared

# best trial:
sortby(Trials)[1]

# result of all / best 100 trials
plot_all_hyppars(plot_hyppar, hyppars, Trials, title="all hyppars")
plot_all_hyppars(plot_hyppar, hyppars, Trials, title="all hyppars", trialmax=100)

# percentage in best trials (all / best 100)
plot_all_hyppars(plot_course_in_best_trials, hyppars, Trials, title="all hyppars")
plot_all_hyppars(plot_course_in_best_trials, hyppars, Trials, title="all hyppars", trialmax=100)

# correlation
plot_correlation(:prep_vars, :prep_K, Trials)
plot_correlation(:hl_dim, :nb_hl, Trials)
plot_correlation(:initializer_weight, :initializer_bias, Trials)
plot_correlation(:lera, :lambda, Trials)
plot_correlation(:beta1, :beta2, Trials)
plot_correlation(:act_fct, :batchsize, Trials, yscale=:log10)
plot_correlation(:act_fct, :initializer_bias, Trials)


# mean result
get_mean_result(:hl_dim, Trials)
get_mean_result(:nb_hl, Trials)
get_mean_result(:batchsize, Trials)
get_mean_result(:initializer_weight, Trials)
get_mean_result(:initializer_bias, Trials)
get_mean_result(:prep_vars, Trials)
get_mean_result(:prep_K, Trials)

# count in best 100 trials
get_count_in_best_trials(:hl_dim, Trials, trialmax=100)
get_count_in_best_trials(:nb_hl, Trials, trialmax=100)
get_count_in_best_trials(:batchsize, Trials, trialmax=100)
get_count_in_best_trials(:initializer_weight, Trials, trialmax=100)
get_count_in_best_trials(:initializer_bias, Trials, trialmax=100)
get_count_in_best_trials(:prep_vars, Trials, trialmax=100)
get_count_in_best_trials(:prep_K, Trials, trialmax=100)


# loss distribution for parameter choice
hist_loss(:act_fct, Trials, trialmax=100)

# occurence in best 100 trials
hist_occurance(:act_fct, Trials, trialmax=100)

# correlation occurence in best 100 trials
hist_correlation_occurance(:lera, :prep_K, Trials, trialmax=100)

