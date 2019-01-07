struct BatchBernoulli{T}
    p::T
end

"""
    logpdf(ab::BatchBernoulli, x)

Compute ``Ber(x; p)``.

NOTE: `x` is assumed to be in batch.
"""
function logpdf(ab::BatchBernoulli, x)
    FT = eltype(ab.p)
    _one = one(FT)
    _eps = eps(FT)
    lp = x .* log.(ab.p .+ _eps) .+ (_one .- x) .* log.(_one + _eps .- ab.p)
    return lp
end

"""
    kl(ab1::BatchBernoulli, ab2::BatchBernoulli)

Compute ``KL(Ber_1||Ber_2)``.
"""
function kl(ab1::BatchBernoulli, ab2::BatchBernoulli)
    FT = eltype(ab1.p)
    if eltype(ab2.p) != FT
        @warn "FT are different for ab1 and ab2" eltype(ab1.p) eltype(ab2.p)
    end
    _one = one(FT)
    _eps = eps(FT)
    kl = ab1.p .* (log.(ab1.p .+ _eps) .- log.(ab2.p .+ _eps)) .+
         (_one .- ab1.p) .* (log.(_one + _eps .- ab1.p) .- log.(_one + _eps .- ab2.p))
    return kl
end

struct BatchBernoulliLogit{T}
    logitp::T
end

"""
    logpdf(ab::BatchBernoulliLogit, x)

Compute ``Ber(x; p)``.

NOTE: `x` is assumed to be in batch.
"""
function logpdf(abl::BatchBernoulliLogit, x)
    FT = eltype(abl.logitp)
    _one = one(FT)
    lp = x .* abl.logitp .- log.(_one .+ exp.(abl.logitp))
    return lp
end
