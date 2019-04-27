function softplus(x)
    return log(one(x) + exp(x))
end

function softplus_safe(x)
    return softplus(x) + eps(FT)
end

function leaky_relu(x; alpha=0.2)
    neg = min(0, x) * alpha
    pos = max(0, x)
    return neg + pos
end
