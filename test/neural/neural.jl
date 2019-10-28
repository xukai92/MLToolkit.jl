using Test
using MLToolkit: include_list_as_module

@testset "Neural" begin
    @warn "`Neural.jl` is not tested."

    tests = [
        "architecture.jl",
    ]

    include_list_as_module(tests, "TestNeural")
end
