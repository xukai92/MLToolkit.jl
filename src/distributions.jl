import Distributions: logpdf, pdf, rand, mode

"""
    mckl(logpdfp, logpdfq, x; args...)

Compute the MC estimate of KL divergence between `p` and `q`.
"""
function mckl(p, q, x; args...)
    return mean(logpdf(p, x; args...) - logpdf(q, x; args...); dims=2)
end
mckl(p, q; args...) = mckl(p, q, rand(p); args...)

include("distributions/displaced_poisson.jl")
include("distributions/ibp.jl")
include("distributions/normal.jl")
include("distributions/gumbel.jl")
include("distributions/bernoulli.jl")
include("distributions/beta.jl")

export logpdf, pdf, rand, mckl, kl, mode
