function softplus(x)
    return log(one(x) + exp(x))
end

function leaky_relu(x; alpha=0.2)
    neg = min(0, x) * alpha
    pos = max(0, x)
    return neg + pos
end
