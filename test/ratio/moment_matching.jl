using Test, Distributions
using MLToolkit: pairwise_dot, pairwise_dot_kai, plt, estimate_r_de, get_r_hat_numerically

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

@testset "demo" begin
    dist_nu = Normal(1, 1)
    dist_de = Normal(0, 2)

    function plotdist!(; ax=plt.gca())
        x = range(-5; stop=5, length=100)
        lp_nu = logpdf.(Ref(dist_nu), x)
        lp_de = logpdf.(Ref(dist_de), x)
        ax.plot(x, exp.(lp_nu), "--", c="blue", label="nu")
        ax.plot(x, exp.(lp_de), "--", c="orange", label="de")
        ax.plot(x, exp.(lp_nu - lp_de), "--", c="green", label="r")
    end

    n_nu = 100
    x_nu = rand(dist_nu, 1, n_nu)
    n_de = 200
    x_de = rand(dist_de, 1, n_de)

    function plotdata!(; ax=plt.gca())
        ax.scatter(x_nu[1,:], zeros(n_nu) .- 0.5, marker="x", c="blue", label="nu", alpha=0.2)
        ax.scatter(x_de[1,:], zeros(n_de) .- 1.0, marker="x", c="orange", label="de", alpha=0.2)
    end

    fig = plt.figure(figsize=(8, 6))

    plotdist!()
    plotdata!()

    r_hat_numerical_00 = estimate_r_de(x_de, x_nu; get_r_hat=get_r_hat_numerically, positive=false, normalisation=false)
    plt.scatter(vec(x_de), r_hat_numerical_00, s=5.0, alpha=0.5, label="r_hat_numerical_00")

    r_hat_numerical_01 = estimate_r_de(x_de, x_nu; get_r_hat=get_r_hat_numerically, positive=false, normalisation=true)
    plt.scatter(vec(x_de), r_hat_numerical_01, s=5.0, alpha=0.5, label="r_hat_numerical_01")

    r_hat_numerical_10 = estimate_r_de(x_de, x_nu; get_r_hat=get_r_hat_numerically, positive=true, normalisation=false)
    plt.scatter(vec(x_de), r_hat_numerical_10, s=5.0, alpha=0.5, label="r_hat_numerical_10")

    r_hat_numerical_11 = estimate_r_de(x_de, x_nu; get_r_hat=get_r_hat_numerically, positive=true, normalisation=true)
    plt.scatter(vec(x_de), r_hat_numerical_11, s=5.0, alpha=0.5, label="r_hat_numerical_11")

    # r_hat_analytical = estimate_r_de(x_de, x_nu; ϵ=0)
    # plt.scatter(vec(x_de), r_hat_analytical, s=5.0, alpha=0.5, label="r_hat_analytical")

    r_hat_analytical_stable = estimate_r_de(x_de, x_nu)
    plt.scatter(vec(x_de), r_hat_analytical_stable, s=5.0, alpha=0.5, label="r_hat_analytical_stable")

    plt.legend()

    fig.savefig("$(@__DIR__)/mmd_ratio_est.png")
end