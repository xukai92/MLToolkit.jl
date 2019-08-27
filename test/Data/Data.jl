using Test
using MLToolkit.Data

@testset "Data" begin
    # @testset "MNIST" begin
    #     tr_sz = 100
    #     te_sz = 200
    #     for mnist_sym = [:mnist, :fmnist]
    #         x_tr_sub, y_tr_sub, x_te_sub, y_te_sub = load_mnist(mnist_sym, tr_sz, te_sz; flatten=true)
    #         @test size(x_tr_sub) == (784, tr_sz)
    #         @test size(y_tr_sub) == (tr_sz,)
    #         @test size(x_te_sub) == (784, te_sz)
    #         @test size(y_te_sub) == (te_sz,)

    #         x_tr_sub, y_tr_sub, x_te_sub, y_te_sub = load_mnist(mnist_sym, tr_sz, te_sz; flatten=false)
    #         @test size(x_tr_sub) == (28, 28, 1, tr_sz)
    #         @test size(y_tr_sub) == (tr_sz,)
    #         @test size(x_te_sub) == (28, 28, 1, te_sz)
    #         @test size(y_te_sub) == (te_sz,)
    #     end
    # end

    @testset "make_imggrid" begin
        d = 5
        rand_img = rand(d^2, 100)
        n_rows = n_cols = 3
        gap = 1
        imggrid = make_imggrid(rand_img, n_rows, n_cols)
        @test all(imggrid[1,:] .== 0.5)
        @test all(imggrid[end,:] .== 0.5)
        @test all(imggrid[:,1] .== 0.5)
        @test all(imggrid[:,end] .== 0.5)
        @test size(imggrid) == (n_rows * d + gap * (n_rows + 1),
                                n_cols * d + gap * (n_cols + 1))
    end

    @testset "BatchDataLoader" begin
        sz = 100
        x = randn(784, sz)
        y = rand(sz)

        batch_size = 20
        mnist_loader = BatchDataLoader(batch_size, x, y; drop_last=true)
        x1, y1 = first(mnist_loader)
        @test length(mnist_loader) == 5
        @test size(x1, 2) == batch_size
        @test size(y1, 1) == batch_size

        mnist_loader = BatchDataLoader(batch_size, x, y; drop_last=false)
        @test length(mnist_loader) == 5

        batch_size = 30

        mnist_loader = BatchDataLoader(batch_size, x, y; drop_last=false)
        @test length(mnist_loader) == 4

        mnist_loader = BatchDataLoader(batch_size, x, y; drop_last=true)
        @test length(mnist_loader) == 3

        at_list = [Array{Float16,2}, Array{Float32,2}, Array{Float64,2}]
        # if AT == KnetArray
            # push!(at_list, AT{FT,2})
        # end
        for at in at_list
            mnist_loader = BatchDataLoader(batch_size, x; drop_last=true, atype=at)
            x1 = first(mnist_loader)
            @test typeof(x1) == at
        end

        @warn "`shuffle!()` not tested"
    end

    @testset "get_features" begin
        @warn "`get_features_griffiths2011indian()` not tested"
        @warn "`get_features_large()` not tested"
    end
end
