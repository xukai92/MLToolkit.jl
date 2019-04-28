using SpecialFunctions: lgamma
import SpecialFunctions: lbeta, beta

# Call back functions for `lbeta` and `beta`
# Those in `SpecialFunctions.jl` are implemented for `(::Number, ::Number)` only,
# for which `KnetArray` cannot broadcast through.
# NOTE: `_lbeta` and `_beta` are implemented for explictly test purpose.
# TODO: implement this broadcasting in Knet.jl
_lbeta(α, β) = lgamma.(α) + lgamma.(β) - lgamma.(α + β)
_beta(α, β) = exp.(_lbeta(α, β))
lbeta(α, β) = _lbeta(α, β)
beta(α, β) = _beta(α, β)

import StatsFuns: logit

function logit(x)
    _eps = eps(FT)
    _one = one(FT)
    return log(x + _eps) - log(_one - x + _eps)
end
