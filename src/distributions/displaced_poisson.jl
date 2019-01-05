using Distributions: DiscreteUnivariateDistribution
using SpecialFunctions: gamma
using GSL: sf_gamma_inc_P
incomplete_gamma = sf_gamma_inc_P

"""
The displaced Poisson distribution (a.k.a the hyper-Poisson distribution).

Ref: https://www.jstor.org/stable/2283992
"""
struct DisplacedPoisson <: DiscreteUnivariateDistribution
    λ::Float64
    r::Float64
    function DisplacedPoisson(λ::Float64, r::Float64)
        @assert λ > 0 "λ is not positve"
        @assert r >= 0 "r is not non-negative"
        return new(λ, r)
    end
end

function pdf(dp::DisplacedPoisson, k::Int)
    if k < dp.r
        return 0
    else
        p = dp.λ^(k - dp.r) * exp(-dp.λ) / gamma(k - dp.r + 1)
        diff = ceil(dp.r) - dp.r
        # incomplete_gamma(a, b) is undefined for a = 0
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

function mode(dp::DisplacedPoisson)
    k = 0
    p = pdf(dp, k)
    while true
        p_next = pdf(dp, k + 1)
        if p_next < p
            break
        end
        p = p_next
        k += 1
    end
    return k
end
