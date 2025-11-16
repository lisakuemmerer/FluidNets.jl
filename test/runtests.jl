# Smoke test for FluidNets
# This test simply checks the package can be loaded and lists exported names.
using Test

@testset "FluidNets smoke" begin
    try
        using FluidNets
        println("FluidNets loaded. Exported names:")
        println(names(FluidNets, all=false))
        @test true
    catch e
        println("Failed to load FluidNets: ", e)
        @test false
    end
end
