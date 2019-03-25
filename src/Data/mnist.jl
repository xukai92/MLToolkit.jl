module MNIST
    import Knet
    include(Knet.dir("data","mnist.jl"))
    export mnist
end
using .MNIST: mnist

module FMNIST
    import Knet
    include(Knet.dir("data","fashion-mnist.jl"))
    export fmnist
end
using .FMNIST: fmnist


"""
    load_mnist(mnist_sym::Symbol, tr_sz::Integer, te_sz::Integer; flatten::Bool=true)

Load a subset of MNIST-like dataset.
"""
function load_mnist(mnist_sym::Symbol, tr_sz::Integer=60_000, te_sz::Integer=10_000;
                    flatten::Bool=true)
    x_tr, y_tr, x_te, y_te = eval(mnist_sym)()

    if flatten
        x_tr_sub = Matrix{FT}(undef, 28 * 28, tr_sz)
        x_te_sub = Matrix{FT}(undef, 28 * 28, te_sz)

        for i = 1:tr_sz
            x_tr_sub[:,i] = reshape(x_tr[:,:,1,i]', 28 * 28, 1)
        end

        for i = 1:te_sz
            x_te_sub[:,i] = reshape(x_te[:,:,1,i]', 28 * 28, 1)
        end
    else
        x_tr_sub = Array{FT,4}(x_tr[:,:,:,1:tr_sz])
        x_te_sub = Array{FT,4}(x_te[:,:,:,1:te_sz])
    end

    y_tr_sub = Vector{FT}(y_tr)[1:tr_sz]
    y_te_sub = Vector{FT}(y_te)[1:te_sz]

    return x_tr_sub, y_tr_sub, x_te_sub, y_te_sub
end
