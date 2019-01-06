"""
The Gumbel-Softmax distributions.

NOTE: parameters are in batch.

Ref: https://arxiv.org/abs/1611.01144
"""
struct GumbelSoftmax{T}
    p::T
end

"""
    rand(gs::GumbelSoftmax{T}; τ=0.1)

Sample from the Gumbel-Softmax distributions.
"""
function rand(gs::GumbelSoftmax{AT}; τ=0.1) where {AT}
    FT = eltype(gs.p)

    u = rand(FT, size(gs.p)...)
    g = AT(-log.(-log.(u)))

    logit = g .+ log.(gs.p)
    exp_logit = exp.(logit ./ τ)
    return exp_logit ./ sum(exp_logit; dims=1)
end
