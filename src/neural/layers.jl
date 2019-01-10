"""
Dense layer
"""
struct Dense <: StaticLayer
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

struct DynamicOut <: StaticLayer
    rnn::Knet.RNN
    mlp::StaticLayer
end

function DynamicOut(i_dim::Integer, h_dim::Integer; rnnType=:relu, f=identity)
    rnn = Knet.RNN(i_dim, h_dim; rnnType=rnnType, dataType=FT)
    mlp = Dense(h_dim, 1; f=f)
    return DynamicOut(rnn, mlp)
end

function (dy::DynamicOut)(x, d::Integer)
    (x_dim, batch_size) = size(x)
    h = dy.rnn(reshape(hcat([x for _ = 1:d]...), x_dim, batch_size, d))
    # The `reshape` below was double-checked - don't waste time on debugging it
    y = dy.mlp(reshape(h, size(h, 1), batch_size * d))
    return reshape(y, batch_size, d)'
end

struct DynamicIn <: StaticLayer
    rnn::Knet.RNN
    mlp::StaticLayer
end

function DynamicIn(h_dim::Integer, o_dim::Integer; rnnType=:relu, f=identity)
    rnn = Knet.RNN(1, h_dim; rnnType=rnnType, dataType=FT)
    mlp = Dense(h_dim, o_dim; f=f)
    return DynamicIn(rnn, mlp)
end

function (dy::DynamicIn)(x)
    (x_dim, batch_size) = size(x)
    h = dy.rnn(reshape(x', 1, batch_size, x_dim))[:,:,end] # only the last hidden state is used
    return dy.mlp(h)
end

const STATIC_SYM_LIST = [:Dense, :DynamicIn, :DynamicOut]

for static_sym in STATIC_SYM_LIST
    sto_sym = Symbol("Gaussian$static_sym")

    @eval begin
        struct $sto_sym <: StochasticLayer
            μ::StaticLayer
            Σ::StaticLayer
        end

        function $sto_sym(i_dim::Integer, h_dim::Integer)
            return $sto_sym($static_sym(i_dim, h_dim), $static_sym(i_dim, h_dim; f=softplus))
        end

        function (gn::$sto_sym)(x, d::Integer...)
            return BatchNormal(gn.μ(x, d...), gn.Σ(x, d...))
        end
    end
end

for static_sym in STATIC_SYM_LIST
    sto_sym = Symbol("GaussianLogVar$static_sym")

    @eval begin
        struct $sto_sym <: StochasticLayer
            μ::StaticLayer
            logΣ::StaticLayer
        end

        function $sto_sym(i_dim::Integer, z_dim::Integer)
            return $sto_sym($static_sym(i_dim, z_dim), $static_sym(i_dim, z_dim))
        end

        function (glvn::$sto_sym)(x, d::Integer...)
            return BatchNormalLogVar(glvn.μ(x, d...), glvn.logΣ(x, d...))
        end
    end
end

for static_sym in STATIC_SYM_LIST
    sto_sym = Symbol("Bernoulli$static_sym")

    @eval begin
        struct $sto_sym <: StochasticLayer
            p::StaticLayer
        end

        function $sto_sym(i_dim::Integer, z_dim::Integer)
            return $sto_sym($static_sym(i_dim, z_dim; f=Knet.sigm))
        end

        function (bn::$sto_sym)(x, d::Integer...)
            return BatchBernoulli(bn.p(x, d...))
        end
    end
end

for static_sym in STATIC_SYM_LIST
    sto_sym = Symbol("BernoulliLogit$static_sym")

    @eval begin
        struct $sto_sym <: StochasticLayer
            logitp::StaticLayer
        end

        function $sto_sym(i_dim::Integer, z_dim::Integer)
            return $sto_sym($static_sym(i_dim, z_dim))
        end

        function (bln::$sto_sym)(x, d::Integer...)
            return BatchBernoulliLogit(bln.logitp(x, d...))
        end
    end
end

"""
Chaining multiple layers.

NOTE: only chainning layers are allowed but not models. As models are assumed to output loss when being called.
"""
struct Chain <: AbstractLayer
    layers::Tuple
    function Chain(layers::Tuple)
        n = length(layers)
        for i = 1:n-1
            @assert layers[i] isa StaticLayer "The layers in middle should be `StaticLayer`."
        end
        @assert layers[n] isa AbstractLayer "The last layer should be a `AbstractLayer`"
        return new(layers)
    end
end
Chain(layers::AbstractLayer...) = Chain(layers)
Chain(layers) = Chain(layers...)

"""
Run chained layers.
"""
function (c::Chain)(x)
    for l in c.layers
        x = l(x)
    end
    return x
end

"""
Run chained layers with `args...` applied to the last one.
"""
function (c::Chain)(x, args...)
    n = length(c.layers)
    for i = 1:n-1
        x = c.layers[i](x)
    end
    return c.layers[n](x, args...)
end
