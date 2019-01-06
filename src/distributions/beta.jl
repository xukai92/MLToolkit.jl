# Sampling functions

"""
    sample_from_kumaraswamy(a, b)

Sample from Kumaraswamy distribution.

NOTE: `a` and `b` are assumed to be in batch

Ref: https://arxiv.org/abs/1605.06197
"""
function sample_from_kumaraswamy(a, b)

    d, n = size(a)

    u = AT(rand(FT, d, n))

    x = (one(FT) .- u.^(one(FT) ./ b)).^(one(FT) ./ a)

    return x

end

function sample_log_from_kumaraswamy(a, b)

    d, n = size(a)
    u = AT(rand(FT, d, n))
    logx = log.(one(FT) .- exp.(log.(u) ./ (b .+ eps(FT))) .+ eps(FT)) ./ (a .+ eps(FT))
    return logx
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

"""
    cond_kl_bernoulli(x, p, q)

Compute ``KL(Ber(x;p)||Ber(x;q))``.

NOTE: `x`, `p` and `q` are assumed to be in batch

NOTE: this function is not tested

Ref: https://github.com/rachtsingh/ibp_vae/blob/3b76ed15e0d7479423f893404c8549246e93c13f/src/training/mf_concrete.py#L36-L38
"""
function cond_kl_bernoulli(x, p, q)

    kl = (log.(p .+ eps(FT)) .* x .+ log.(one(FT) .- p .+ eps(FT)) .* (one(FT) .- x)) .-
         (log.(q .+ eps(FT)) .* x .+ log.(one(FT) .- q .+ eps(FT)) .* (one(FT) .- x))

    return kl

end

"""
    cond_kl_bernoulli_by_logit(logitx, logitp1, logitp2; τ1=FT(0.1), τ2=FT(0.1))

Compute ``KL(ExpConcrete(x;p1)||ExpConcrete(x;p2))`` using Monte Carlo.

NOTE: `logitx`, `logitp1` and `logitp2` are assumed to be in batch

Ref: https://github.com/rachtsingh/ibp_vae/blob/3b76ed15e0d7479423f893404c8549246e93c13f/src/training/common.py#L57-L66
"""
function cond_kl_bernoulli_by_logit(logitx, logitp1, logitp2; τ1=FT(0.1), τ2=FT(0.1))

    lp1 = logpdf_bernoulli_by_logit(logitp1, logitx; τ=τ1)
    lp2 = logpdf_bernoulli_by_logit(logitp2, logitx; τ=τ2)

    kl = lp1 - lp2

    return kl

end

# Log-pdf functions


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
