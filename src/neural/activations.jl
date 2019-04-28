# TODO: implement CUDA kernels for below

log1pexp(x) = log1p(exp(x))
logexpm1(x) = log(-expm1(x))

const softplus = log1pexp
const invsoftplus = logexpm1

function leaky_relu(x; alpha=0.2)
    neg = min(0, x) * alpha
    pos = max(0, x)
    return neg + pos
end
