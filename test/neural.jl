using Distributed, Test

@testset "Neural" begin

    @warn "`neural.jl` is not tested."

    tests = [
        "nodes",
        "layers",
        "activations",
    ]

    res = map(tests) do t
        @eval module $(Symbol("TestNeural_", t))
        include("neural/" * $t * ".jl")
        end
        return
    end
end
