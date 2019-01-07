"""
Dense layer
"""
struct Dense <: AbstractTrainable
    w
    b
    f::Function
end

function (d::Dense)(x)
    return d.f.(d.w * Knet.mat(x) .+ d.b)
end

"""
Create dense layer by input and output size.
"""
function Dense(i_dim::Integer, o_dim::Integer; f::Function=identity)
    w = Knet.param(o_dim, i_dim; atype=AT{FT,2})
    b = Knet.param0(o_dim; atype=AT{FT,1})
    return Dense(w, b, f)
end

# Stochastic nodes

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

struct GaussianLogVarNode <: AbstractTrainable
    f::AbstractTrainable
end

function GaussianLogVarNode(i_dim::Integer, z_dim::Integer)
    return GaussianLogVarNode(Dense(i_dim, 2 * z_dim))
end

function (ge::GaussianLogVarNode)(x)
    h = ge.f(x)
    z_dim = round(Integer, size(h, 1) / 2)

    return BatchNormalLogVar(h[1:z_dim,:], h[z_dim+1:end,:])
end

struct BernoulliNode <: AbstractTrainable
    f::AbstractTrainable
end

function BernoulliNode(i_dim::Integer, z_dim::Integer)
    return BernoulliNode(Dense(i_dim, z_dim))
end

function (ge::BernoulliNode)(x)
    p = Knet.sigm.(ge.f(x))
    return BatchBernoulli(p)
end

struct BernoulliLogitNode <: AbstractTrainable
    f::AbstractTrainable
end

function BernoulliLogitNode(i_dim::Integer, z_dim::Integer)
    return BernoulliLogitNode(Dense(i_dim, z_dim))
end

function (ge::BernoulliLogitNode)(x)
    logitp = ge.f(x)
    return BatchBernoulliLogit(logitp)
end
