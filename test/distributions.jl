using Test

@testset "Distributions" begin
    include("distributions/displaced_poisson.jl")
    include("distributions/ibp.jl")
    include("distributions/normal.jl")
    include("distributions/bernoulli.jl")
    include("distributions/gumbel.jl")
end
