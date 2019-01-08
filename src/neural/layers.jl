"""
Dense layer
"""
struct Dense <: AbstractTrainable
    w
    b
    f::Function
end

function (d::Dense)(x)
    return d.f.(d.w * x .+ d.b)
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
    μ::AbstractTrainable
    Σ::AbstractTrainable
end

function GaussianNode(i_dim::Integer, z_dim::Integer; ltype=Dense)
    return GaussianNode(ltype(i_dim, z_dim), ltype(i_dim, z_dim; f=softplus))
end

function (gn::GaussianNode)(x)
    return BatchNormal(gn.μ(x), gn.Σ(x))
end

struct GaussianLogVarNode <: AbstractTrainable
    μ::AbstractTrainable
    logΣ::AbstractTrainable
end

function GaussianLogVarNode(i_dim::Integer, z_dim::Integer; ltype=Dense)
    return GaussianLogVarNode(ltype(i_dim, z_dim), ltype(i_dim, z_dim))
end

function (glvn::GaussianLogVarNode)(x)
    return BatchNormalLogVar(glvn.μ(x), glvn.logΣ(x))
end

struct BernoulliNode <: AbstractTrainable
    p::AbstractTrainable
end

function BernoulliNode(i_dim::Integer, z_dim::Integer; ltype=Dense)
    return BernoulliNode(ltype(i_dim, z_dim; f=Knet.sigm))
end

function (bn::BernoulliNode)(x)
    return BatchBernoulli(bn.p(x))
end

struct BernoulliLogitNode <: AbstractTrainable
    logitp::AbstractTrainable
end

function BernoulliLogitNode(i_dim::Integer, z_dim::Integer; ltype=Dense)
    return BernoulliLogitNode(ltype(i_dim, z_dim))
end

function (bln::BernoulliLogitNode)(x)
    return BatchBernoulliLogit(bln.logitp(x))
end
