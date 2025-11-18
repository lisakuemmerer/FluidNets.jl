

##############################################################################################################
# minimal use of fluidum to calculate a freeze out


# initialize fluid properties
fluidprop = FluidProperties(FluiduMEoS(), QGPViscosity(0.2,0.2), SimpleBulkViscosity(0.032,15.0), ZeroDiffusion()); 

# define a radial grid with rmax, gridpoints
grid_params = Fluidum.GridParameters(30,300);

# define time span for evolution
tspan = (0.2,15);

# define parameters normalization entropy profile, tau_0, rdrop
initial_params = Fluidum.InitialParameters(90., tspan[1], 8.);

# load entropy ( you will need to specity the correct file)
ent = TabulatedData("/home/lisa/MA/Data/EntropyProfile_ALICE.txt");

# initialize fields on grid for entropy & centrality(min, max)
fields = initialize_fields(ent, 0, 10; grid_params=grid_params, initial_params=initial_params);

# compute freezeout in dependence to the FO temperature
freezeout_results(TFO) = freeze_out_routine(fields.discrete_field,Fluidum.matrxi1d_visc!,fluidprop,fields.initial_field,tspan,Tfo=TFO);

# standard freezeout at TFO = 156.5MeV
freezeout_results_default = freezeout_results(0.1565)





##############################################################################################################
# minimal use of fluidum to calculate a freeze out


function compare_spectra_2D(part, K_func, NN; pt_min=0., pt_max=10., steps=100, comp_ratio=true, plotlog=true)

    t0 = time()
    spectrum_NN = [s[1] for s in spectra(freezeout_results_default, part, NN, fluidprop, pt_min=pt_min,pt_max=pt_max,step=steps)]
    t1 = time()
    spectrum_interpol = [s[1] for s in spectra(freezeout_results_default, part, K_func, fluidprop, pt_min=pt_min,pt_max=pt_max,step=steps)]
    t2 = time()
    println("Time for spectrum calculation:")
    println("NN: ", t1-t0, " Interpol: ", t2-t1)

    spectrum_ratio = spectrum_NN ./spectrum_interpol
    pt = collect(range(pt_min,stop=pt_max,length=steps))
    
    t3 = time()
    multi_interpol = multiplicity(freezeout_results_default, part, K_func, fluidprop, pt_min=pt_min,pt_max=pt_max)
    t4 = time()
    multi_NN = multiplicity(freezeout_results_default, part, NN, fluidprop, pt_min=pt_min,pt_max=pt_max)
    t5 = time()
    println("Time for multiplicity calculation:")
    println("NN: ", t5-t4, " Interpol: ", t4-t3)

    println("Interpolated multiplicity: ", multi_interpol)
    println("NN multiplicity: ", multi_NN)
    
    plot(pt, spectrum_interpol, label="Interpolated", xlabel="pT", ylabel="1/2πpT dN/dpT")
    plot!(pt, spectrum_NN, label="NN")
    comp_ratio && plot!(pt, spectrum_ratio, label="NN / Interpolated", lc=:gray, ls=:dash)

    if plotlog==true
        plot!(yaxis=:log, ylims=(1e-2, 1e4))
    end

    m_I=Int(round(multi_interpol[1]))
    m_I_std=Int(round(multi_interpol[2]))
    m_NN=Int(round(multi_NN[1]))
    m_NN_std=Int(round(multi_NN[2]))
    
    annotate!(0.1, 1e-1, text(String("N_I=$m_I" * "±" * "$m_I_std\n" * "N_NN=$m_NN" * "±" * "$m_NN_std"), :left, 10))
    plot!()
end


function compare_spectra_4D(Tc, Tk, part, K_func, NN; pt_min=0., pt_max=10., steps=100, comp_ratio=true, plotlog=true)

    FO = freezeout_results(Tk)

    t0 = time()
    spectrum_NN = [s[1] for s in spectra(FO, part, NN, Tc, fluidprop, pt_min=pt_min,pt_max=pt_max,step=steps)]
    t1 = time()
    spectrum_interpol = [s[1] for s in spectra(FO, part, K_func, Tc, fluidprop, pt_min=pt_min,pt_max=pt_max,step=steps)]
    t2 = time()
    println("Time for spectrum calculation:")
    println("NN: ", t1-t0, " Interpol: ", t2-t1)

    spectrum_ratio = spectrum_NN ./spectrum_interpol
    pt = collect(range(pt_min,stop=pt_max,length=steps))
    
    t3 = time()
    multi_interpol = multiplicity(FO, part, K_func, Tc, fluidprop, pt_min=pt_min,pt_max=pt_max)
    t4 = time()
    multi_NN = multiplicity(FO, part, NN, Tc, fluidprop, pt_min=pt_min,pt_max=pt_max)
    t5 = time()
    println("Time for multiplicity calculation:")
    println("NN: ", t5-t4, " Interpol: ", t4-t3)

    println("Interpolated multiplicity: ", multi_interpol)
    println("NN multiplicity: ", multi_NN)
    
    plot(pt, spectrum_interpol, label="Interpolated", xlabel="pT", ylabel="1/2πpT dN/dpT")
    plot!(pt, spectrum_NN, label="NN")
    comp_ratio && plot!(pt, spectrum_ratio, label="NN / Interpolated", lc=:gray, ls=:dash)

    if plotlog==true
        plot!(yaxis=:log, ylims=(1e-2, 1e4))
    end

    m_I=Int(round(multi_interpol[1]))
    m_I_std=Int(round(multi_interpol[2]))
    m_NN=Int(round(multi_NN[1]))
    m_NN_std=Int(round(multi_NN[2]))
    
    annotate!(0.1, 1e-1, text(String("N_I=$m_I" * "±" * "$m_I_std\n" * "N_NN=$m_NN" * "±" * "$m_NN_std"), :left, 10))
    plot!()
end

export compare_spectra_2D, compare_spectra_4D