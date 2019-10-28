using Test
using MLToolkit: include_list_as_module

@testset "Neural" begin
    tests = [
        "architecture.jl",
    ]

    include_list_as_module(tests, "TestNeural")

    @warn "`Neural.jl` is not tested."

    @warn "`gen.jl` is not tested."
end
