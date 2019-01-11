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
    i_dim::Integer
    h_dim::Integer
    rnn::Knet.RNN
    mlp::StaticLayer
end

function DynamicOut(i_dim::Integer, h_dim::Integer; rnnType=:relu, f=identity)
    rnn = Knet.RNN(i_dim, h_dim; rnnType=rnnType, dataType=FT)
    mlp = Dense(h_dim, 1; f=f)
    return DynamicOut(rnn, mlp)
end

function (dy::DynamicOut)(x, d::Integer=dy.h_dim)
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

const UNARY_DIST_DICT = Dict(
    :Bernoulli => (:BatchBernoulli, :p, :(Knet.sigm)),
    :BernoulliLogit => (:BatchBernoulliLogit, :logitp, :identity),
    :GumbelBernoulliLogit => (:BatchGumbelBernoulliLogit, :logitp, :identity),
)

for dist_sym in keys(UNARY_DIST_DICT)
    (batch_dist_sym, field_sym, f_sym) = UNARY_DIST_DICT[dist_sym]
    for static_sym in STATIC_SYM_LIST
        sto_sym = Symbol("$dist_sym$static_sym")

        @eval begin
            struct $sto_sym <: StochasticLayer
                $field_sym::StaticLayer
            end

            function $sto_sym(i_dim::Integer, z_dim::Integer)
                return $sto_sym($static_sym(i_dim, z_dim; f=$f_sym))
            end

            function (sto::$sto_sym)(x, d::Integer...)
                return $batch_dist_sym(sto.$field_sym(x, d...))
            end
        end
    end
end

const BINARY_DIST_DICT = Dict(
    :Gaussian => (:BatchNormal, :μ, :identity,
                                :Σ, :softplus),
    :GaussianLogVar => (:BatchNormalLogVar, :μ, :identity,
                                            :logΣ, :identity),
    :Kumaraswamy => (:BatchKumaraswamy, :a, :softplus,
                                        :b, :softplus),
)

for dist_sym in keys(BINARY_DIST_DICT)
    (batch_dist_sym, field_sym_1, f_sym_1,
                     field_sym_2, f_sym_2) = BINARY_DIST_DICT[dist_sym]
    for static_sym in STATIC_SYM_LIST
        sto_sym = Symbol("$dist_sym$static_sym")

        @eval begin
            struct $sto_sym <: StochasticLayer
                $field_sym_1::StaticLayer
                $field_sym_2::StaticLayer
            end

            function $sto_sym(i_dim::Integer, z_dim::Integer)
                return $sto_sym($static_sym(i_dim, z_dim; f=$f_sym_1),
                                $static_sym(i_dim, z_dim; f=$f_sym_2))
            end

            function (sto::$sto_sym)(x, d::Integer...)
                return $batch_dist_sym(sto.$field_sym_1(x, d...),
                                       sto.$field_sym_2(x, d...))
            end
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
