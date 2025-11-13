

##############################################################################################################
# this script allows to get files containing BG & separate modes respectively from the FastReso files
# output files have content in form of ROWS for pt,ur,Tchem,Tkin,kernels(BG or mode)
# content exactly as used in the further functions (read_data a.s.o.) and can be used directly for the network training


# THIS SCRIPT ASSUMES THAT:

# 1. you have FastReso output files with names in form of "PDGid_2212_total_T0.1100_Tchem_0.1300_Kj.out" 
# (which would be a file containing Kernels for the proton (PDGid_2212), containing resonances(total), for Tkin=110MeV & Tchem=130MeV)
# if this is not the case, you will need to adjust the function below so it reads the correct files

# 2. the file content is structured in columns for pT, ur, & kernels as named in K_labels in order
# if this is not the case, you need to restructure the function below
# OR consider which kernels you are treating in the further Network structures (Training as well as treatment in FluiduM)

# 3. pT & ur values are on a grid (each combination is contained), the grid is the same throughout all the files
# if this is not the case, the function below will still work as it is supposed to (output in corresponding pT,uR,Tchem,Tkin,kernels as rows)
# but the variable values will not rise in order(pT first, then ur, ...), but be mixed in the order they were read in
# this should in general not be a problem, but maybe needs to be considered when using data as train/test sets


# adjust this to how particles are specified in filename. this will also influence how output is saved thrpughout the whole module
particle_ids = Dict{}(:pion=>"PDGid_211", :proton=>"PDGid_2212", :kaon=>"PDGid_321")


# name of the labels IN ORDER as in files
K_labels = ["Keq1", "Keq2", "Kshear1", "Kshear2", "Kshear3", "Kshear4", "Kbulk1", "Kbulk2", 

    "Ktemp1,0", "Ktemp2,0",
    "Ktemp1,1", "Ktemp2,1",
    "Ktemp1,2", "Ktemp2,2",
    "Ktemp1,3", "Ktemp2,3",
    "Ktemp1,4", "Ktemp2,4", 

    "Kvel1,0", "Kvel2,0", "Kvel3,0", "Kvel4,0",
    "Kvel1,1", "Kvel2,1", "Kvel3,1", "Kvel4,1", 
    "Kvel1,2", "Kvel2,2", "Kvel3,2", "Kvel4,2",
    "Kvel1,3", "Kvel2,3", "Kvel3,3", "Kvel4,3", 
    "Kvel1,4", "Kvel2,4", "Kvel3,4", "Kvel4,4", 

    "Ksheartemp1,0", "Ksheartemp2,0", "Ksheartemp3,0", "Ksheartemp4,0",
    "Ksheartemp1,1", "Ksheartemp2,1", "Ksheartemp3,1", "Ksheartemp4,1", 
    "Ksheartemp1,2", "Ksheartemp2,2", "Ksheartemp3,2", "Ksheartemp4,2",
    "Ksheartemp1,3", "Ksheartemp2,3", "Ksheartemp3,3", "Ksheartemp4,3", 
    "Ksheartemp1,4", "Ksheartemp2,4", "Ksheartemp3,4", "Ksheartemp4,4", 
    
    "Kshearshear1,0", "Kshearshear2,0", "Kshearshear3,0", "Kshearshear4,0", "Kshearshear5,0", "Kshearshear6,0", "Kshearshear7,0", "Kshearshear8,0", "Kshearshear9,0", "Kshearshear10,0", "Kshearshear11,0", "Kshearshear12,0", "Kshearshear13,0", "Kshearshear14,0", "Kshearshear15,0", "Kshearshear16,0", "Kshearshear17,0", "Kshearshear18,0", "Kshearshear19,0", "Kshearshear20,0", "Kshearshear21,0", "Kshearshear22,0", 
    "Kshearshear1,1", "Kshearshear2,1", "Kshearshear3,1", "Kshearshear4,1", "Kshearshear5,1", "Kshearshear6,1", "Kshearshear7,1", "Kshearshear8,1", "Kshearshear9,1", "Kshearshear10,1", "Kshearshear11,1", "Kshearshear12,1", "Kshearshear13,1", "Kshearshear14,1", "Kshearshear15,1", "Kshearshear16,1", "Kshearshear17,1", "Kshearshear18,1", "Kshearshear19,1", "Kshearshear20,1", "Kshearshear21,1", "Kshearshear22,1", 
    "Kshearshear1,2", "Kshearshear2,2", "Kshearshear3,2", "Kshearshear4,2", "Kshearshear5,2", "Kshearshear6,2", "Kshearshear7,2", "Kshearshear8,2", "Kshearshear9,2", "Kshearshear10,2", "Kshearshear11,2", "Kshearshear12,2", "Kshearshear13,2", "Kshearshear14,2", "Kshearshear15,2", "Kshearshear16,2", "Kshearshear17,2", "Kshearshear18,2", "Kshearshear19,2", "Kshearshear20,2", "Kshearshear21,2", "Kshearshear22,2", 
    "Kshearshear1,3", "Kshearshear2,3", "Kshearshear3,3", "Kshearshear4,3", "Kshearshear5,3", "Kshearshear6,3", "Kshearshear7,3", "Kshearshear8,3", "Kshearshear9,3", "Kshearshear10,3", "Kshearshear11,3", "Kshearshear12,3", "Kshearshear13,3", "Kshearshear14,3", "Kshearshear15,3", "Kshearshear16,3", "Kshearshear17,3", "Kshearshear18,3", "Kshearshear19,3", "Kshearshear20,3", "Kshearshear21,3", "Kshearshear22,3", 
    "Kshearshear1,4", "Kshearshear2,4", "Kshearshear3,4", "Kshearshear4,4", "Kshearshear5,4", "Kshearshear6,4", "Kshearshear7,4", "Kshearshear8,4", "Kshearshear9,4", "Kshearshear10,4", "Kshearshear11,4", "Kshearshear12,4", "Kshearshear13,4", "Kshearshear14,4", "Kshearshear15,4", "Kshearshear16,4", "Kshearshear17,4", "Kshearshear18,4", "Kshearshear19,4", "Kshearshear20,4", "Kshearshear21,4", "Kshearshear22,4", 

    "Kshearvel1,0", "Kshearvel2,0", "Kshearvel3,0", "Kshearvel4,0", "Kshearvel5,0", "Kshearvel6,0", "Kshearvel7,0", "Kshearvel8,0", "Kshearvel9,0", "Kshearvel10,0", "Kshearvel11,0", "Kshearvel12,0", 
    "Kshearvel1,1", "Kshearvel2,1", "Kshearvel3,1", "Kshearvel4,1", "Kshearvel5,1", "Kshearvel6,1", "Kshearvel7,1", "Kshearvel8,1", "Kshearvel9,1", "Kshearvel10,1", "Kshearvel11,1", "Kshearvel12,1", 
    "Kshearvel1,2", "Kshearvel2,2", "Kshearvel3,2", "Kshearvel4,2", "Kshearvel5,2", "Kshearvel6,2", "Kshearvel7,2", "Kshearvel8,2", "Kshearvel9,2", "Kshearvel10,2", "Kshearvel11,2", "Kshearvel12,2", 
    "Kshearvel1,3", "Kshearvel2,3", "Kshearvel3,3", "Kshearvel4,3", "Kshearvel5,3", "Kshearvel6,3", "Kshearvel7,3", "Kshearvel8,3", "Kshearvel9,3", "Kshearvel10,3", "Kshearvel11,3", "Kshearvel12,3", 
    "Kshearvel1,4", "Kshearvel2,4", "Kshearvel3,4", "Kshearvel4,4", "Kshearvel5,4", "Kshearvel6,4", "Kshearvel7,4", "Kshearvel8,4", "Kshearvel9,4", "Kshearvel10,4", "Kshearvel11,4", "Kshearvel12,4", 
    
    "Kbulktemp1,0", "Kbulktemp2,0",
    "Kbulktemp1,1", "Kbulktemp2,1",
    "Kbulktemp1,2", "Kbulktemp2,2",
    "Kbulktemp1,3", "Kbulktemp2,3",
    "Kbulktemp1,4", "Kbulktemp2,4", 
    
    "Kbulkvel1,0", "Kbulkvel2,0", "Kbulkvel3,0", "Kbulkvel4,0",
    "Kbulkvel1,1", "Kbulkvel2,1", "Kbulkvel3,1", "Kbulkvel4,1", 
    "Kbulkvel1,2", "Kbulkvel2,2", "Kbulkvel3,2", "Kbulkvel4,2",
    "Kbulkvel1,3", "Kbulkvel2,3", "Kbulkvel3,3", "Kbulkvel4,3", 
    "Kbulkvel1,4", "Kbulkvel2,4", "Kbulkvel3,4", "Kbulkvel4,4", 
    
    "Kbulkbulk1,0", "Kbulkbulk2,0",
    "Kbulkbulk1,1", "Kbulkbulk2,1",
    "Kbulkbulk1,2", "Kbulkbulk2,2",
    "Kbulkbulk1,3", "Kbulkbulk2,3",
    "Kbulkbulk1,4", "Kbulkbulk2,4"]

    
#function that returns the indices of Kernels of one m-mode
function modeindex(m)
    return [i[1] for i in findall(contains(",$m"), K_labels)].+2
end



# this function reads files for particle=(:pion,:proton,:kaon), which=("total","thermal") from a directory specified by filepath
# output files (BG and modes 0-4) saved to outpath
function get_BG_mode_files(particle, which, filepath, outpath)

    #make sure this loops over the right directory where the files are
    files = filter(contains(which), filter(contains(particle_ids[particle]), readdir(filepath)))
    
    # find modes depending on K_labels -> check manually with first line in files if these are right !!!
    modeindex = [modeindex(j) for j in 0:4]

    K_BG = []
    K_m0 = []
    K_m1 = []
    K_m2 = []
    K_m3 = []
    K_m4 = []
    for f in files
        println("starting with file ", f)
        d = readdlm(String(filepath*f), Float64, skipstart=1)
        # variables
        pt = transpose(d[:,1])
        ur = transpose(d[:,2])
        tk = parse(Float64, match(r"T([0-9.]+)", f).captures[1]) * ones(size(pt))
        tc = parse(Float64, match(r"Tchem_([0-9.]+)", f).captures[1]) * ones(size(pt))
        # background kernels
        BG = transpose(d[:,3:10])
        # mode kernels
        m0 = transpose(d[:,modeindex[1]])
        m1 = transpose(d[:,modeindex[2]])
        m2 = transpose(d[:,modeindex[3]])
        m3 = transpose(d[:,modeindex[4]])
        m4 = transpose(d[:,modeindex[5]])
        # fill arrays
        push!(K_BG, vcat(pt,ur,tc,tk,BG))
        push!(K_m0, vcat(pt,ur,tc,tk,m0))
        push!(K_m1, vcat(pt,ur,tc,tk,m1))
        push!(K_m2, vcat(pt,ur,tc,tk,m2))
        push!(K_m3, vcat(pt,ur,tc,tk,m3))
        push!(K_m4, vcat(pt,ur,tc,tk,m4))
    end

    # save files to some other directory
    println("saving files")
    writedlm(String(outpath*String(particle)*"_"*which*"_BG.txt"), hcat(K_BG...))
    writedlm(String(outpath*String(particle)*"_"*which*"_m0.txt"), hcat(K_m0...))
    writedlm(String(outpath*String(particle)*"_"*which*"_m1.txt"), hcat(K_m1...))
    writedlm(String(outpath*String(particle)*"_"*which*"_m2.txt"), hcat(K_m2...))
    writedlm(String(outpath*String(particle)*"_"*which*"_m3.txt"), hcat(K_m3...))
    writedlm(String(outpath*String(particle)*"_"*which*"_m4.txt"), hcat(K_m4...))
end


# # example call of function on my structures to get files for pion with resonances --- RUNS FOR LIKE 1h !!!
# get_BG_mode_files(:pion, "total", "/home/lisa/MA/Data/Full_PCE/Kernels/", "/home/lisa/MA/Data/Full_PCE/modewise/")






##############################################################################################################
# sheartemp weird stuff:
# I ran into files that did not contain m=(1-4) modes for the sheartemp kernels due to an computational error
# -> needed to combine the read-in with files that contained only BG and sheartemp kernels
# in general, no one should ever need this again, but if you run into similar problems this can be used as approach


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
    

#get_BG_m0_files(:proton, "thermal")
#get_mode_files(:proton)


