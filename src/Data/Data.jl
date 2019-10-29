module Data

import Random, Distributions, MLDataUtils
using Distributions: rand, logpdf

include("features.jl")
export get_features_griffiths2011indian, get_features_large

struct Ring{T<:AbstractFloat} <: Distributions.ContinuousMultivariateDistribution
    n_clusters::Int
    s::Int
    σ::T
end

function makemixturemodel(ring::Ring)
    π_typed = convert(typeof(ring.σ), π)
    cluster_indices = collect(0:ring.n_clusters-1)
    base_angle = π_typed * 2 / ring.n_clusters
    angle = (base_angle .* cluster_indices) .- π_typed / 2
    μ = [ring.s * cos.(angle) ring.s * sin.(angle)]'
    return Distributions.MixtureModel([Distributions.MvNormal(μ[:,i], ring.σ) for i in 1:size(μ, 2)])
end

Distributions.rand(rng::Random.AbstractRNG, ring::Ring{T}, n::Int) where {T} = convert.(T, rand(rng, makemixturemodel(ring), n))

Distributions.logpdf(ring::Ring, x::AbstractArray{<:AbstractFloat,2}) = logpdf(makemixturemodel(ring), x)

export Ring, rand, logpdf

### Dataset and data loader

## Dataset

struct Dataset{T}
    name::String
    train::T
    test
    validation
    function Dataset(train::T, test=nothing, validation=nothing; name::String="") where {T}
        for set in (train, test, validation)
            if !isnothing(set)
                if set isa Tuple
                    # Length consistency
                    @assert reduce(==, map(t -> last(size(t)), set))
                    # Dim consistency
                    @assert train isa Tuple # if `set` is `Tuple`, `train` must be
                    @assert length(train) == length(set)
                    for i in 1:length(train)
                        @assert Base.front(size(train[i])) == Base.front(size(set[i]))
                    end
                else
                    # Dim consistency
                    @assert Base.front(size(train)) == Base.front(size(set))
                end
            end
        end
        return new{T}(name, train, test, validation)
    end
end

_dim(d::Tuple{Int}) = first(d)
_dim(d::Tuple) = d

function Distributions.dim(d::Dataset)
    d = Base.front(size(d.train))
    return _dim(d)
end

function Distributions.dim(d::Dataset{<:Tuple})
    d = Base.front(size(first(d.train)))
    return _dim(d)
end

## Data loader

struct DataLoader
    dataset::Dataset
    batch_size::Int
    batch_size_eval::Int
    function DataLoader(dataset::Dataset, batch_size::Int, batch_size_eval::Int=batch_size)
        return new(dataset, batch_size, batch_size_eval)
    end
end

selectdata(data::AbstractArray{<:Any,1}, idx) = data[idx]
selectdata(data::AbstractArray{<:Any,2}, idx) = data[:,idx]
selectdata(data::AbstractArray{<:Any,3}, idx) = data[:,:,idx]
selectdata(data::AbstractArray{<:Any,4}, idx) = data[:,:,:,idx]
selectdata(data::Tuple, idx) = map(d -> selectdata(d, idx), data)

ndata(data) = last(size(data))
ndata(data::Tuple) = last(size(first(data)))

function Base.getproperty(dl::DataLoader, k::Symbol)
    if k in (:train, :test, :validation)
        data = getproperty(dl.dataset, k)
        n = ndata(data)
        batch_size = k == :train ? dl.batch_size : dl.batch_size_eval
        idx_iterator = Iterators.partition(MLDataUtils.shuffleobs(1:n), batch_size)
        return ((data=selectdata(data, idx), idx=idx) for idx in idx_iterator)
    else
        getfield(dl, k)
    end
end

export Dataset, dim, DataLoader

end # module
