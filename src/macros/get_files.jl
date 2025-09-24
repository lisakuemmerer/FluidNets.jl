

particle_ids = Dict{}(:pion=>"PDGid_211", :proton=>"PDGid_2212", :kaon=>"PDGid_321")



# file names in form of "PDGid_2212_total_T0.1100_Tchem_0.1300_Kj.out"
# if you have files with pt, ur, kernels in form of K_labels
function get_BG_mode_files(particle, which)

    #make sure this loops over the right directory where the files are
    files = filter(contains(which), filter(contains(particle_ids[particle]), readdir("/home/lisa/MA/Data/Full_PCE/Kernels/")))
    
    # find modes depending on K_labels -> check manually with first line in files if these are right !!!
    modeindex = [[i[1] for i in findall(contains(",$j"), K_labels)].+2 for j in 0:4]

    K_BG = []
    K_m0 = []
    K_m1 = []
    K_m2 = []
    K_m3 = []
    K_m4 = []
    for f in files
        println("starting with file ", f)
        d = readdlm(String("/home/lisa/MA/Data/Full_PCE/Kernels/"*f), Float64, skipstart=1)
        # variables
        pt = transpose(d[:,1])
        ur = transpose(d[:,2])
        tk = parse(Float64, match(r"T([0-9.]+)", f).captures[1]) * ones(size(pt))
        tc = parse(Float64, match(r"Tchem_([0-9.]+)", f).captures[1]) * ones(size(pt))
        #background kernels
        BG = transpose(d[:,3:10])
        #mode kernels
        m0 = transpose(d[:,modeindex[1]])
        m1 = transpose(d[:,modeindex[2]])
        m2 = transpose(d[:,modeindex[3]])
        m3 = transpose(d[:,modeindex[4]])
        m4 = transpose(d[:,modeindex[5]])
        #fill arrays
        push!(K_BG, vcat(pt,ur,tc,tk,BG))
        push!(K_m0, vcat(pt,ur,tc,tk,m0))
        push!(K_m1, vcat(pt,ur,tc,tk,m1))
        push!(K_m2, vcat(pt,ur,tc,tk,m2))
        push!(K_m3, vcat(pt,ur,tc,tk,m3))
        push!(K_m4, vcat(pt,ur,tc,tk,m4))
    end

    # save files to some other directory
    println("saving files")
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_BG.txt"), hcat(K_BG...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_m0.txt"), hcat(K_m0...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_m1.txt"), hcat(K_m1...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_m2.txt"), hcat(K_m2...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_m3.txt"), hcat(K_m3...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_m4.txt"), hcat(K_m4...))
end



############################################################################################################################################################################
# for sheartemp weird stuff:

function get_BG_m0_files(particle, which)

    files = filter(contains(which), filter(contains(particle_ids[particle]), readdir("/home/lisa/MA/Data/Full_PCE/Kernels/")))
    
    modeindex = [i[1] for i in findall(contains(",0"), K_labels)].+2
    modeindex_data = copy(modeindex)
    for i in eachindex(modeindex)
        modeindex_data[i] = modeindex[i] <= 44 ? modeindex[i] : modeindex[i] .- 16 
    end

    K_BG = []
    K_m0 = []
    for f in files
        println("starting with file ", f)
        d = readdlm(String("/home/lisa/MA/Data/Full_PCE/Kernels/"*f), Float64, skipstart=1)
        pt = transpose(d[:,1])
        ur = transpose(d[:,2])
        tk = parse(Float64, match(r"T([0-9.]+)", f).captures[1]) * ones(size(pt))
        tc = parse(Float64, match(r"Tchem_([0-9.]+)", f).captures[1]) * ones(size(pt))
        BG = transpose(d[:,3:10])
        m0 = transpose(d[:,modeindex_data])
        push!(K_BG, vcat(pt,ur,tc,tk,BG))
        push!(K_m0, vcat(pt,ur,tc,tk,m0))
    end

    println("saving files")
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_BG.txt"), hcat(K_BG...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_"*which*"_m0.txt"), hcat(K_m0...))
end

#get_BG_m0_files(:proton, "thermal")


#########################################

function get_mode_files(particle)
    
    files = filter(contains("total"), filter(contains(particle_ids[particle]), readdir("/home/lisa/MA/Data/Full_PCE/Kernels/")))
    #check temps same
    files != filter(contains("total"), filter(contains(particle_ids[particle]), readdir("/home/lisa/MA/Data/Full_PCE/shearTempKers/"))) && @error "Temperature grid not equal!!!"

    modeindex = [[i[1] for i in findall(contains(",$j"), K_labels)] for j in 1:4]

    K_m1 = []
    K_m2 = []
    K_m3 = []
    K_m4 = []
    for f in files
        println("starting with file ", f)
        d = readdlm(String("/home/lisa/MA/Data/Full_PCE/Kernels/"*f), Float64, skipstart=1)
        d_st = readdlm(String("/home/lisa/MA/Data/Full_PCE/shearTempKers/"*f), Float64, skipstart=1)

        #check pt ur grid same
        d[:,1:2] != d_st[:,1:2] && @error "pt ur grid not equal"

        pt = transpose(d[:,1])
        ur = transpose(d[:,2])
        tk = parse(Float64, match(r"T([0-9.]+)", f).captures[1]) * ones(size(pt))
        tc = parse(Float64, match(r"Tchem_([0-9.]+)", f).captures[1]) * ones(size(pt))

        #check sheartemp m0 ks are te same
        d[:,41:44] != d_st[:,3:6] && @error "Ksheartemp not equal"

        K_beg = d[:,3:44]
        K_st_ms = d_st[:,7:end]
        K_end = d[:,45:end]
        K_full = hcat(K_beg, K_st_ms, K_end)

        push!(K_m1, vcat(pt,ur,tc,tk,transpose(K_full[:,modeindex[1]])))
        push!(K_m2, vcat(pt,ur,tc,tk,transpose(K_full[:,modeindex[2]])))
        push!(K_m3, vcat(pt,ur,tc,tk,transpose(K_full[:,modeindex[3]])))
        push!(K_m4, vcat(pt,ur,tc,tk,transpose(K_full[:,modeindex[4]])))
    end

    println("saving files")
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_total_m1.txt"), hcat(K_m1...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_total_m2.txt"), hcat(K_m2...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_total_m3.txt"), hcat(K_m3...))
    writedlm(String("/home/lisa/MA/Data/Full_PCE/modewise/"*String(particle)*"_total_m4.txt"), hcat(K_m4...))
end
    

#get_mode_files(:proton)



#################################################################################################################################################################################
# find what files I have:

# filter Files in FullPCE

allfiles = readdir("/home/lisa/MA/Data/Full_PCE/shearTempKers/")

# pion, proton: 1612 files, kaon: 64 files (no on sheartemp)
pionfiles = filter(contains("PDGid_211"), allfiles)
protonfiles = filter(contains("PDGid_2212"), allfiles)
kaonfiles = filter(contains("PDGid_321"), allfiles)
length(pionfiles) + length(protonfiles) + length(kaonfiles)


# FullPCE: pion&proton: thermal & total each 806, kaon: 64 thermal, no total
# sheartemp: 806 total each
files = protonfiles
total = filter(contains("total"), files)
thermal = filter(contains("thermal"), files)
length(total) + length(thermal)


# FullPCE&sheartemp: total&thermal, pion&proton: 26 Tchems, 31 Tkins, 
whichfiles = total
T_kin = unique([parse(Float64, match(r"T([0-9.]+)", f).captures[1]) for f in whichfiles])
T_chem = unique([parse(Float64, match(r"Tchem_([0-9.]+)", f).captures[1]) for f in whichfiles])
length(T_kin) * length(T_chem)
# FullPCE kaon thermal: 26 Tchems, 3 Tkins
k = [(parse(Float64, match(r"T([0-9.]+)", f).captures[1]), parse(Float64, match(r"Tchem_([0-9.]+)", f).captures[1])) for f in whichfiles]
[length(filter(t->t[1]==tk, k)) for tk in unique([t[1] for t in k])]

###################################################
#whats in files:


f = readdlm("/home/lisa/MA/Data/Full_PCE/Kernels/PDGid_2212_total_T0.1100_Tchem_0.1300_Kj.out")

#ptur grid
pts = f[2:end,1]
pt = unique(pts)
urs = f[2:end,2]
ur = unique(urs)
length(pt)*length(ur)

# FullPCE: 252 kernels + pt,ur
# sheartemp: 20 Kernels + pt,ur
filter(x->x!="", f[2,:])

# FullPCE: 
# sheartemp
print(k for k in f[1,2:end])
filter(x->occursin(":", x), filter(x->x isa SubString{String}, f[1,2:end]))







##########################################################################################################################################################################################
# weird kernels ...different Ks, alsways some missing on pT, uR grid ---> ??????????????????

###################
#Kj.3out

f = readdlm("/home/lisa/MA/Data/Full_PCE/weird_kernels/PDGid_211_thermal_T0.1100_Tchem_0.1400_Kj.3out")

#vars : pt ur grid, some missing ????
pts = f[2:end,1]
pt = unique(pts)
urs = f[2:end,2]
ur = unique(urs)

unique([length(filter(x->x==pt[i], pts)) for i in eachindex(pt)])
unique([length(filter(x->x==ur[i], urs)) for i in eachindex(ur)])


# 18 kernels + pt,ur
filter(x->x!="", f[2,:])

# BG , temp 12, m in (0,4)
println(k for k in f[1,2:end])

#########################
#Kj2.out


f = readdlm("/home/lisa/MA/Data/Full_PCE/weird_kernels/PDGid_211_thermal_T0.1100_Tchem_0.1400_Kj4.out")

# ks same as Kernelfiles
println(k for k in f[1,2:end])
filter(x->occursin(":", x), filter(x->x isa SubString{String}, f[1,2:end]))

# 18 kernels + pt,ur
filter(x->x!="", f[2,:])


pts = f[2:end,1]
pt = unique(pts)
urs = f[2:end,2]
ur = unique(urs)

unique([length(filter(x->x==pt[i], pts)) for i in eachindex(pt)])
unique([length(filter(x->x==ur[i], urs)) for i in eachindex(ur)])




