using Test

@testset "Tests" begin
    include("data.jl")
    include("special.jl")
    include("activations.jl")
    include("distributions.jl")
    include("transformations.jl")
end
