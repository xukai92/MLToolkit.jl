using Knet: param, param0, mat

"""
Dense layer
"""
struct Dense <: AbstractTrainable
    w
    b
    f::Function
end

function (d::Dense)(x)
    return d.f.(d.w * mat(x) .+ d.b)
end

"""
Create dense layer by input and output size.
"""
function Dense(i_dim::Integer, o_dim::Integer; f::Function=identity)
    w = param(o_dim, i_dim; atype=AT{FT,2})
    b = param0(o_dim; atype=AT{FT,1})
    return Dense(w, b, f)
end

"""
Chaining multiple layers.
"""
struct Chain <: AbstractTrainable
    layers
end

function (c::Chain)(x)
    for l in c.layers
        x = l(x)
    end
    return x
end