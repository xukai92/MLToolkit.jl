using Test, MLToolkit

@testset "IBP" begin
    d, n = 10, 5
    for _ = 1:NUM_RANDTESTS
        ibp = IBP(2.0)
        Z = rand(ibp, d, n)
        @test size(Z) == (n, d)
        Z = rand(ibp, n)
        @test size(Z, 1) == n
    end
end
