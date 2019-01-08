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

struct DynamicOut <: AbstractTrainable
    rnn
    mlp::AbstractTrainable
end

function DynamicOut(i_dim::Integer, h_dim::Integer; rnnType=:relu, f=identity)
    rnn = RNN(i_dim, h_dim; rnnType=rnnType, dataType=FT)
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

struct DynamicIn <: AbstractTrainable
    rnn
    mlp::AbstractTrainable
end

function DynamicIn(h_dim::Integer, o_dim::Integer; rnnType=:relu, f=identity)
    rnn = RNN(1, h_dim; rnnType=rnnType, dataType=FT)
    mlp = Dense(h_dim, o_dim; f=f)
    return DynamicIn(rnn, mlp)
end

function (dy::DynamicIn)(x)
    (x_dim, batch_size) = size(x)
    h = dy.rnn(reshape(x', 1, batch_size, x_dim))[:,:,end] # only the last hidden state is used
    return dy.mlp(h)
end
