using Test, MLToolkit
using Knet
include(Knet.dir("data","mnist.jl"))
include(Knet.dir("data","fashion-mnist.jl"))

@testset "Data" begin
    @testset "MNIST" begin
        tr_sz = 100
        te_sz = 200
        for mf = [mnist, fmnist]
            x_tr_sub, y_tr_sub, x_te_sub, y_te_sub = load_mnist(mnist, tr_sz, te_sz; flatten=true)
            @test size(x_tr_sub) == (784, tr_sz)
            @test size(y_tr_sub) == (tr_sz,)
            @test size(x_te_sub) == (784, te_sz)
            @test size(y_te_sub) == (te_sz,)

            x_tr_sub, y_tr_sub, x_te_sub, y_te_sub = load_mnist(fmnist, tr_sz, te_sz; flatten=false)
            @test size(x_tr_sub) == (28, 28, 1, tr_sz)
            @test size(y_tr_sub) == (tr_sz,)
            @test size(x_te_sub) == (28, 28, 1, te_sz)
            @test size(y_te_sub) == (te_sz,)
        end
    end
end
