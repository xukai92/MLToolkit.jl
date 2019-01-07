function exp_softplus(x)
    return one(x) + exp(x)
end

function softplus(x)
    return log(exp_softplus(x))
end

function leaky_relu(x; alpha=0.2)
    neg = min(0, x) * alpha
    pos = max(0, x)
    return neg + pos
end
