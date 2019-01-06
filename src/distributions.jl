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
export DisplacedPoisson

include("distributions/ibp.jl")
export IBP

include("distributions/normal.jl")
export UnivariateNormal, BatchNormal

include("distributions/gumbel.jl")
export GumbelSoftmax, GumbelSoftmax2D, GumbelBernoulli, GumbelBernoulliLogit

include("distributions/bernoulli.jl")
export BatchBernoulli

export logpdf, pdf, rand, mckl, kl, mode
