using MLToolkit, Test

@testset "IBP" begin
    d, n = 10, 5

    ibp = IBP(2.0)
    Z = rand(ibp, d, n)
    @test size(Z) == (n, d)
end
