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
A stochastic node for the Gaussian distribution. It returns `BatchNormal`.
"""
struct GaussianNode <: AbstractTrainable
    f::AbstractTrainable
end

function GaussianNode(i_dim::Integer, z_dim::Integer)
    return GaussianNode(Dense(i_dim, 2 * z_dim))
end

function (ge::GaussianNode)(x)
    h = ge.f(x)
    z_dim = round(Integer, size(h, 1) / 2)

    return BatchNormal(h[1:z_dim,:], softplus.(h[z_dim+1:end,:]))
end

"""
A stochastic node for the Bernoulli distribution. It returns `BatchBernoulliLogit`.
"""
struct BernoulliNode <: AbstractTrainable
    f::AbstractTrainable
end

function BernoulliNode(i_dim::Integer, z_dim::Integer)
    return BernoulliNode(Dense(i_dim, z_dim))
end

function (ge::BernoulliNode)(x)
    logitp = ge.f(x)
    return BatchBernoulliLogit(logitp)
end
