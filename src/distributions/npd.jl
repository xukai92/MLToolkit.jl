using StatsFuns: logistic
using Base: setindex!, getindex

"""
    Nonparametric discrete distribution with parameters in logit space.

This represents a discrete distribution with p.m.f of

```math
P(K = k) = (1 - \\rho_k) \\prod_{i = 1}^{k - 1} \\rho_i.
```
"""
struct LogitNPD{T<:Real,A<:AbstractVector{T}} <: Distributions.DiscreteUnivariateDistribution
    logitρ  ::  A
end
# TODO: find a way to declare initial value

LogitNPD() = LogitNPD(0)
LogitNPD(alpha::AbstractFloat) = LogitNPD(ceil(Int, alpha))
LogitNPD(k_init::Int) = LogitNPD(ones(FT, k_init) * 0.0)

function getlogitρ(lnpd::LogitNPD{T,A}, k::Int) where {T<:Real,A<:AbstractVector{T}}
    l = length(lnpd.logitρ)
    if k > l
        append!(lnpd.logitρ, ones(T, k - l) * 0.0...)
    end
    return lnpd.logitρ[k]
end

function getlogitρ(lnpd::LogitNPD{T,A}, k1::Int, k2::Int) where {T<:Real,A<:AbstractVector{T}}
    l = length(lnpd.logitρ)
    if k2 > l
        append!(lnpd.logitρ, ones(T, k2 - l) * 0.0...)
    end
    return lnpd.logitρ[k1:k2]
end

getρ(lnpd::LogitNPD, k...) = logistic.(getlogitρ(lnpd, k...))

function getlogρ(lnpd::LogitNPD, k...)
    l = getlogitρ(lnpd, k...)
    return l .- log1pexp.(l)
end

function pdf(lnpd::LogitNPD, k::Int)
    if k <= 0
        return 0
    elseif k == 1
        return 1 - getρ(lnpd, 1)
    else
        return prod(getρ(lnpd, 1, k - 1)) * (1 - getρ(lnpd, k))
    end
end

function logpdf(lnpd::LogitNPD, k::Int)
    if k <= 0
        return -Inf
    elseif k == 1
        return log(1 - getρ(lnpd, 1))
    else
        return sum(getlogrho(lnpd, 1, k - 1)) + log(1 - getρ(lnpd, k))
    end
end

function ccdf(lnpd::LogitNPD{T,A}, k::Int) where {T<:Real,A<:AbstractVector{T}}
    if k <= 0
        return one(T)
    else
        return prod(getρ(lnpd, 1, k))
    end
end

function cdf(lnpd::LogitNPD, k::Int)
    return 1 - ccdf(lnpd, k)
end

function rand(lnpd::LogitNPD{T,A}) where {T<:Real,A<:AbstractVector{T}}
    i = 1
    while true
        u = rand(T)
        if u > getρ(lnpd, i)
            return i
        end
        i = i + 1
    end
end

function rand(lnpd::LogitNPD{T,A}, n::Int) where {T<:Real,A<:AbstractVector{T}}
    return Int[rand(lnpd) for _ = 1:n]
end

function mode(lnpd::LogitNPD)
    p = pdf(lnpd, 1)
    p_acc = p_max = p
    k = k_max = 1
    while p_max < 1 - p_acc
        p_next = pdf(lnpd, k + 1)
        p_acc += p_next
        if p_next > p_max
            p_max = p_next
            k_max = k
        end
        k = k + 1
    end
    return k_max
end
