using Distributions: MvNormal

struct GaussianDataset{D} <: AbstractDataset{D}
    X
    Xt
end

n_display(::GaussianDataset) = 1_000

function GaussianDataset(
    n_data::Int, 
    mean::AbstractVector{T}, 
    cov::AbstractVecOrMat{T}; 
    seed::Int=1,
    test_ratio=1/6,
    n_test::Int=ratio2num(n_data, test_ratio),
) where {T<:AbstractFloat}
    rng = MersenneTwister(seed)
    X = rand(rng, MvNormal(mean, cov), n_data)
    Xt = rand(rng, MvNormal(mean, cov), n_test)
    return GaussianDataset{length(mean)}(X, Xt)
end
