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
function Dense(i_dim::Integer, o_dim::Integer; f::Function=identity, b0::Bool=true)
    w = Knet.param(o_dim, i_dim; atype=AT{FT,2})
    if b0
        b = Knet.param0(o_dim; atype=AT{FT,1})
    else
        b = Knet.param(o_dim; atype=AT{FT,1})
    end
    return Dense(w, b, f)
end

"""
Lazy dense layer ...zZzZzZ
"""
struct LazyDense <: StaticLayer
    w
    b
    f::Function
    b0::Bool
end

function (d::LazyDense)(x, k::Int=length(d.b); b0::Bool=d.b0)
    sx1 = size(x, 1)
    if sx1 > size(d.w, 2)
        d.w.value = hcat([d.w.value, Knet.param(size(d.w, 1), sx1 - size(d.w, 2); atype=AT{FT,2}).value]...)
    end
    if k > size(d.w, 1)
        k_diff = k - size(d.w, 1)
        d.w.value = vcat([d.w.value, Knet.param(k_diff, size(d.w, 2); atype=AT{FT,2}).value]...)
        d.b.value = vcat([d.b.value, (b0 ? Knet.param0 : Knet.param)(k_diff; atype=AT{FT,1}).value]...)
    end
    return d.f.(d.w[1:k,1:sx1] * x .+ d.b[1:k])
end

function LazyDense(i_dim::Integer, o_init::Integer; f::Function=identity, b0::Bool=true)
    d = Dense(i_dim, o_init; f=f, b0=b0)
    return LazyDense(d.w, d.b, d.f, b0)
end

"""
Dynamic output using RNN
"""
struct DynamicOut <: StaticLayer
    rnn::Knet.RNN
    mlp::StaticLayer
end

function DynamicOut(i_dim::Integer, r_dim::Integer;
                    skipInput=false, rnnType=:relu, bidirectional=false, f=identity)
    rnn = Knet.RNN(i_dim, r_dim; skipInput=skipInput, rnnType=rnnType, bidirectional=bidirectional, dataType=FT)
    mlp = Dense(r_dim * (bidirectional ? 2 : 1), 1; f=f)
    return DynamicOut(rnn, mlp)
end

function (dy::DynamicOut)(x, d::Integer=dy.r_dim)
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

"""
Dynamic input using RNN
"""
function DynamicIn(r_dim::Integer, o_dim::Integer;
                   skipInput=false, rnnType=:relu, bidirectional=false, f=identity)
    rnn = Knet.RNN(1, r_dim; skipInput=skipInput, rnnType=rnnType, bidirectional=bidirectional, dataType=FT)
    mlp = Dense(r_dim * (bidirectional ? 2 : 1), o_dim; f=f)
    return DynamicIn(rnn, mlp)
end

function (dy::DynamicIn)(x)
    (x_dim, batch_size) = size(x)
    y = dy.rnn(reshape(x', 1, batch_size, x_dim))
    # Only the last hidden state is used
    h = if dy.rnn.direction == 0
        y[:,:,end]
    else
        h_dim = dy.rnn.hiddenSize
        vcat(y[:,:,end][1:h_dim,:], y[:,:,1][h_dim+1:end,:])
    end
    return dy.mlp(h)
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
