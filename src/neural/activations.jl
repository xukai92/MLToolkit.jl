# TODO: see if I can replace this fucntion with StatsFuns.jl

function softplus(x)
    return log(one(x) + exp(x))
end

function invsoftplus(x)
    return log(exp(x) - one(x))
end

function leaky_relu(x; alpha=0.2)
    neg = min(0, x) * alpha
    pos = max(0, x)
    return neg + pos
end
