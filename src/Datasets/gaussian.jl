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
    rng=MersenneTwister(seed),
) where {T<:AbstractFloat}
    X = rand(rng, MvNormal(mean, cov), n_data)
    Xt = rand(rng, MvNormal(mean, cov), n_test)
    D = length(mean)
    @info "Oh you just get the $D D Gaussian dataset" n_data=n_data n_test=n_test mean=mean cov=cov
    return GaussianDataset{D}(X, Xt)
end

GaussianDataset(n_data; kwargs...) = GaussianDataset(n_data, 2f0 * ones(Float32, 2), [1 81f-2; 81f-2 1])
