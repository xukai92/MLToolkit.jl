using Test
using MLToolkit: pairwise_dot, pairwise_dot_kai

@testset "pairwise_dot()" begin
    xtest = [
        1.0 2.0 4.0; 
        1.0 2.0 4.0
    ]

    ytest = [
        0.0 2.0 18.0;
        2.0 0.0  8.0;
        18.0 8.0  0.0
    ]

    @testset "Correctness of `pairwise_dot_kai(x)`" begin
        n_randtests = 10
        @test pairwise_dot_kai(xtest) == ytest
        for _ = 1:n_randtests
            xrand = randn(784, 100)
            @test pairwise_dot_kai(xrand) ≈ pairwise_dot(xrand)
        end
    end

    @testset "Correctness of `pairwise_dot_kai(x, y)`" begin
        n_randtests = 10
        @test pairwise_dot_kai(xtest) == ytest
        for _ = 1:n_randtests
            xrand = randn(784, 100)
            yrand = randn(784, 200)
            @test pairwise_dot_kai(xrand, yrand) ≈ pairwise_dot(xrand, yrand)
        end
    end

    # xbench = randn(784, 100)
    # @benchmark pairwise_dot(xbench)

    # xbench = xbench |> cu
    # @benchmark pairwise_dot(xbench)
end