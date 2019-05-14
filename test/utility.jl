using Test, MLToolkit

@testset "Utility" begin
    @testset "count_leadingzeros" begin
        @test count_leadingzeros([0,0,1,0]) == 2
        @test count_leadingzeros([0,0,0,0]) == 4
        @test count_leadingzeros([1,0,0,0]) == 0
    end

    @warn `turnoffgpu() not tested`
end
