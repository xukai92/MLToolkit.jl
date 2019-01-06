struct Kumaraswamy{T}
    a::T
    b::T
end

"""
    rand(k::Kumaraswamy{AT}) where {AT}

Sample from Kumaraswamy distribution.

NOTE: `k.a` and `k.b` are assumed to be in batch

Ref: https://arxiv.org/abs/1605.06197
"""
function rand(kuma::Kumaraswamy{AT}) where {AT}
    FT = eltype(kuma.a)
    _one = one(FT)
    u = AT(rand(FT, size(kuma.a)...))
    x = (_one .- u.^(_one ./ kuma.b)).^(_one ./ kuma.a)
    return x
end

function logrand(kuma::Kumaraswamy{AT}) where {AT}
    FT = eltype(kuma.a)
    _one = one(FT)
    _eps = eps(FT)
    u = AT(rand(FT, size(kuma.a)...))
    logx = log.(_one .- exp.(log.(u) ./ (kuma.b .+ _eps)) .+ _eps) ./ (kuma.a .+ _eps)
    return logx
end

export Kumaraswamy



"""
    logpdf_kumaraswamy(a, b, x)

Compute ``Kumaraswamy(x; a, b)``.

NOTE: only `x` is assumed to be in batch

NOTE: this function is not tested
"""
function logpdf_kumaraswamy(a, b, x)

    lp = log(a) .+ log(b) .+ (a - one(FT)) .* log.(x) .+ (b - one(FT)) .* log.(one(FT) .- x.^a)

    return lp

end

"""
    sample_from_kumaraswamy_iid(a, b, d, n)

Sample from Kumaraswamy distribution i.i.d to a `d` by `n` matrix.

Ref: https://arxiv.org/abs/1605.06197
Ref: https://github.com/rachtsingh/ibp_vae/blob/3b76ed15e0d7479423f893404c8549246e93c13f/src/models/common.py
"""
function sample_from_kumaraswamy_iid(a, b, d, n)

    return exp.(sample_log_from_kumaraswamy_iid(a, b, d, n))

end
function sample_log_from_kumaraswamy_iid(a, b, d, n)

    u = AT(rand(FT, d, n))

    # x = (one(FT) .- u.^(one(FT) ./ b)).^(one(FT) ./ a)
    logx = log.(one(FT) .- exp.(log.(u) ./ (b .+ eps(FT))) .+ eps(FT)) ./ (a .+ eps(FT))

    return logx

end



# KL divergence functions

"""
    kl_dirichlet(α, β)

Compute ``KL(Dir(α)||Dir(β))`` where ``α = [α_1, \\dots, α_K]`` and ``β = [β_1, \\dots, β_K]``.

NOTE: this function is not tested

Ref: http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
"""
function kl_dirichlet(α, β)

    α0 = sum(α)
    β0 = sum(β)

    kl = (lgamma.(α0) .- sum(lgamma.(α))) .- (lgamma(β0) - sum(lgamma.(β))) .+
          sum((α .- β) .* (digamma.(α) .- digamma(α0)))

    return kl

end

"""
    kl_beta_slow(α1, β1, α2, β2)

Compute ``KL(Beta(α1, β1)||Beta(α2, β2))``.
"""
kl_beta_slow(α1, β1, α2, β2) = kl_dirichlet([α1, β1], [α2, β2])

"""
    kl_beta(α1, β1, α2, β2)

Compute ``KL(Beta(α1, β1)||Beta(α2, β2))``.

Note: `α1`, `β1`, `α2` and `β2` are assumed to be in batch
"""
function kl_beta(α1, β1, α2, β2)

    α0 = α1 .+ β1
    β0 = α2 .+ β2

    kl = (lgamma.(α0) .- (lgamma.(α1) .+ lgamma.(β1))) .- (lgamma.(β0) .- (lgamma.(α2) .+ lgamma.(β2))) .+
         (α1 - α2) .* (digamma.(α1) - digamma.(α0)) .+ (β1 - β2) .* (digamma.(β1) - digamma.(α0))

end

"""
    kl_kumaraswamy_beta(a, b, α, β; M=10)

Compute ``KL(Kumaraswamy(a, b)||Beta(α, β))``.

NOTE: only `a` and `b` are assumed to be in batch

NOTE: this function is not tested
"""
function kl_kumaraswamy_beta(a, b, α, β; M=10)

    @assert M > 0
    acc = one(FT) ./ (FT(1.0) .+ a .* b) .* beta(FT(1.0) ./ a, b)
    for m = 2:M
        acc = acc .+ one(FT) ./ (FT(m) .+ a .* b) .* beta(FT(m) ./ a, b)
    end

    kl = (a .- α) ./ a .* (-FT(Base.MathConstants.γ) .- digamma.(b) .- one(FT) ./ b) .+ log.(a .* b) .+
         lbeta(α, β) .- (b .- one(FT)) ./ b .+ (β - one(FT)) .* b .* acc

end

# Log-pdf functions
