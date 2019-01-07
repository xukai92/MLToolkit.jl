using SpecialFunctions: lgamma
import SpecialFunctions: lbeta, beta

# Call back functions for `lbeta` and `beta`
# Those in `SpecialFunctions.jl` are implemented for `(::Number, ::Number)` only,
# for which `KnetArray` cannot broadcast through.
# NOTE: `_lbeta` and `_beta` are implemented for explictly test purpose.
_lbeta(α, β) = lgamma(α) + lgamma(β) - lgamma(α + β)
_beta(α, β) = exp(_lbeta(α, β))
lbeta(α, β) = _lbeta(α, β)
beta(α, β) = _beta(α, β)
