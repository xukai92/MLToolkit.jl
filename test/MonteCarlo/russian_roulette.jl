using Test, MLToolkit
using Distributions: Poisson

@testset "Russian Roulette" begin
    function X(i)
        if i < 0
            return 0
        end
        if i > 10
            return 0
        end
        return i <= 5 ? i : 10 - i
    end

    S_true = sum(X.(1:10))

    dist = Poisson(5.0)

    n_mc = 100_000
    S_rr = mean([roll(X, dist) for _ = 1:n_mc])

    @test S_rr ≈ S_true atol=0.1
    @test roll(X, dist, n_mc) ≈ S_true atol=0.1

    dist = Poisson(5.0)
    T(i) = pdf(dist, i) * X(i)

    S_true = sum(X.(1:10) .* pdf.(Ref(dist), 1:10))

    n_mc = 100_000
    S_rr_1 = mean([roll(T, dist) for _ = 1:n_mc])
    S_rr_2 = mean([roll_expectation(X, dist, Poisson(5.1)) for _ = 1:n_mc])
    S_rr_3 = mean([roll_expectation(X, dist, LogitNPD()) for _ = 1:n_mc])

    @test S_rr_1 ≈ S_true atol=0.1
    @test S_rr_2 ≈ S_true atol=0.1
    @test S_rr_3 ≈ S_true atol=0.1

    dist = LogitNPD()
    T(i) = pdf(dist, i) * X(i)

    S_true = sum(X.(1:10) .* pdf.(Ref(dist), 1:10))

    n_mc = 100_000
    S_rr_1 = mean([roll(T, dist) for _ = 1:n_mc])
    S_rr_2 = mean([roll_expectation(X, dist, Poisson(5.1)) for _ = 1:n_mc])
    S_rr_3 = mean([roll_expectation(X, dist, LogitNPD()) for _ = 1:n_mc])
    S_rr_4 = mean([roll_expectation(X, dist, dist) for _ = 1:n_mc])

    @test S_rr_1 ≈ S_true atol=0.1
    @test S_rr_2 ≈ S_true atol=0.1
    @test S_rr_3 ≈ S_true atol=0.1
    @test S_rr_4 ≈ S_true atol=0.1
end
