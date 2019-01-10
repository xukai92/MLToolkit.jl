struct BatchBernoulli{T}
    p::T
end

"""
    logpdf(bb::BatchBernoulli, x)

Compute ``Ber(x; p)``.

NOTE: `x` is assumed to be in batch.
"""
function logpdf(bb::BatchBernoulli, x)
    FT = eltype(bb.p)
    _one = one(FT)
    _eps = eps(FT)
    lp = x .* log.(bb.p .+ _eps) .+ (_one .- x) .* log.(_one + _eps .- bb.p)
    return lp
end

mean(bb::BatchBernoulli) = bb.p
mode(bb::BatchBernoulli) = (bb.p .> eltype(bb.p)(0.5)) .* 1

"""
    kldiv(ab1::BatchBernoulli, ab2::BatchBernoulli)

Compute ``KL(Ber_1||Ber_2)``.
"""
function kldiv(ab1::BatchBernoulli, ab2::BatchBernoulli)
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
    logpdf(bb::BatchBernoulliLogit, x)

Compute ``Ber(x; p)``.

NOTE: `x` is assumed to be in batch.
"""
function logpdf(bbl::BatchBernoulliLogit, x)
    FT = eltype(bbl.logitp)
    _one = one(FT)
    lp = x .* bbl.logitp .- log.(_one .+ exp.(bbl.logitp))
    return lp
end

mean(bbl::BatchBernoulliLogit) = Knet.sigm.(bbl.logitp)
function mode(bbl::BatchBernoulliLogit)
    p = Knet.sigm.(bbl.logitp)
    return (p .> eltype(p)(0.5)) .* 1
end
