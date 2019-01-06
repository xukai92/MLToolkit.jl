using Test

@testset "Tests" begin
    include("special.jl")
    include("activations.jl")
    include("distributions.jl")
    include("transformations.jl")
end
