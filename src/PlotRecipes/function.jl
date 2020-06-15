struct ContinuousFunction{D,T<:Function}
    f::T
end

ContinuousFunction{D}(f::T) where {D,T} = ContinuousFunction{D,T}(f)

(func::ContinuousFunction)(args...; kwargs...) = func.f(args...; kwargs...)

### OneDimFunction

OneDimFunction(f::T) where {T} = ContinuousFunction{1}(f::T)

@recipe function f(func::ContinuousFunction{1,F}, start, stop; length=100) where {F}
    label --> :none

    x = range(start, stop; length=length)
    y = map(xi -> func(xi), x)
    x, y
end

### TwoDimFunction

TwoDimFunction(f::T) where {T} = ContinuousFunction{2}(f::T)

@recipe function f(func::ContinuousFunction{2}, xstart, xstop, ystart, ystop; length=100)
    label --> :none
    
    xbins = range(xstart, xstop; length=length)
    ybins = range(ystart, ystop; length=length)
    xygrid = [[xy...] for xy in Iterators.product(xbins, ybins)]
    zgrid = func(hcat(xygrid[:]...))
    zgrid = reshape(zgrid, size(xygrid)...)
    xbins, ybins, zgrid
end

### Distributions

using Distributions: ContinuousUnivariateDistribution, ContinuousMultivariateDistribution, pdf

function default_range(dist::ContinuousUnivariateDistribution, n::Int=3)
    μ, σ = mean(dist), std(dist)
    return (μ - n * σ, μ + n * σ)
end

@recipe function f(dist::T, args...) where {T<:ContinuousUnivariateDistribution}
    start, stop = isempty(args) ? default_range(dist) : args
    ContinuousFunction{1}(x -> pdf(dist, x)), start, stop
end

function default_range(dist::ContinuousMultivariateDistribution, n::Int=3)
    μ, σ = mean(dist), sqrt.(var(dist))
    start, stop = μ - n * σ, μ + n * σ
    return (start[1], stop[1], start[2], stop[2])
end

@recipe function f(dist::T, args...) where {T<:ContinuousMultivariateDistribution}
    xstart, xstop, ystart, ystop = isempty(args) ? default_range(dist) : args
    ContinuousFunction{2}(x -> pdf(dist, x)), xstart, xstop, ystart, ystop
end
