using Distributions: MvNormal

struct GaussianDataset <: AbstractDataset
    X
end

n_display(::GaussianDataset) = 1_000

function GaussianDataset(n_data::Int, mean::T, cov::T; seed::Int=1) where {T}
    rng = MersenneTwister(seed)
    X = rand(rng, MvNormal(mean * ones(T, 2), [1 cov; cov 1]), n_data)
    return GaussianDataset(X)
end
