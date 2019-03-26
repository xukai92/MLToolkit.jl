module MonteCarlo

using StatsBase: counts
using Distributions: DiscreteUnivariateDistribution, pdf, ccdf
const DiscreteUniDist = DiscreteUnivariateDistribution
using ..MLToolkit: LogitNPD, getρ
include("russian_roulette.jl")
export roll, roll_expectation

using Statistics: mean, var

"""
    estimator_stats(estimates; times=nothing, bias=nothing, verbosity=false)

Compute the statistics for a Monte Carlo estimator, given a sequence of repeated estimates and their corresponding running times.
"""
function estimator_stats(estimates::AbstractVector{T};
                         times=nothing, ground=nothing) where {T<:Real}
    s = Dict{Symbol,T}()
    s[:mean] = mean(estimates)
    s[:variance] = var(estimates)
    if times != nothing
        s[:efficiency] = 1 / (s[:variance] * mean(times))
    end
    if ground != nothing
        s[:bias] = abs(s[:mean] - ground)
    end
    return s
end

export estimator_stats

end # module
