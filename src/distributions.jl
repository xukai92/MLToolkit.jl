using SpecialFunctions: gamma
using GSL: sf_gamma_inc_P
using Distributions: DiscreteUnivariateDistribution
import Distributions: pdf, rand

incomplete_gamma = sf_gamma_inc_P

"""
The displaced Poisson distribution (a.k.a the hyper-Poisson distribution).

Ref: https://www.jstor.org/stable/2283992
"""
struct DisplacedPoisson <: DiscreteUnivariateDistribution
    λ::Float64
    r::Float64
end

function pdf(dp::DisplacedPoisson, k::Int)
    if k < dp.r
        return 0
    else
        p = dp.λ^(k - dp.r) * exp(-dp.λ) / gamma(k - dp.r + 1)
        diff = ceil(dp.r) - dp.r
        if !iszero(diff)
            return p / incomplete_gamma(diff, dp.λ)
        else
            return p
        end
    end
end

function rand(dp::DisplacedPoisson)
    u = rand()
    cm = 0
    k = -1
    while cm < u
        k += 1
        cm += pdf(dp, k)
    end
    return k
end
