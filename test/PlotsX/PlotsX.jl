using Test, MLToolkit.PlotsX, Plots
pyplot()
theme(:ggplot2; size=(400, 300))

@testset "Plots" begin
    @testset "autoget_lims" begin
        x = [0 1; 0 1]
        xlims, ylims, zlims = autoget_lims(x)
        @test xlims == [-0.1, 1.1]
        @test ylims == [-0.1, 1.1]
        @test zlims == nothing

        x = [0 1; 0 1; 0 1]
        xlims, ylims, zlims = autoget_lims(x)
        @test xlims == [-0.1, 1.1]
        @test ylims == [-0.1, 1.1]
        @test zlims == [-0.1, 1.1]
    end

    @testset "make_imggrid" begin
        d = 5
        rand_img = rand(d, d, 100)
        n_rows = n_cols = 3
        gap = 1
        imggrid = make_imggrid(rand_img, n_rows, n_cols)
        @test all(imggrid[1,:] .== 0.5)
        @test all(imggrid[end,:] .== 0.5)
        @test all(imggrid[:,1] .== 0.5)
        @test all(imggrid[:,end] .== 0.5)
        @test size(imggrid) == (
            n_rows * d + gap * (n_rows + 1),
            n_cols * d + gap * (n_cols + 1)
        )
    end

    @testset "count_leadingzeros" begin
        @test count_leadingzeros([0, 0, 1]) == 2
    end

    function test_save(p, fname; do_savetex=true)
        save(joinpath(@__DIR__, "$fname.png"), p)
        do_savetex && save(joinpath(@__DIR__, "$fname.tex"), p)
    end
    
    @testset "OneDimFunction" begin
        using Distributions
        dist = Normal(0, 1)
        
        p = plot(dist)
        
        test_save(p, "OneDimFunction")
    end
    
    @testset "ErrorBarLines" begin
        x = 1:10
        ys = [sin.(x) + rand(10) for i in 1:3]
        lines = ErrorBarLines(x, ys)
        
        p = plot(lines; nstd=2, label="Noisy sin(x) with 2 std", legend=:topright)
    
        test_save(p, "ErrorBarLines")
    end
    
    @testset "TwoYAxesLines" begin
        x = collect(1:0.5:10)
        y1 = sin.(x)
        y2 = x .^ 2
    
        lines = TwoYAxesLines(x, y1, y2)
        p = plot(lines; xlabel="x", ylabel1="sin(x)", ylabel2="x^2")
    
        test_save(p, "TwoYAxesLines")
    end
    
    @testset "TwoDimFunction" begin
        using Distributions
        dist = MvNormal(zeros(2), 1)
        
        p = plot(dist)
    
        test_save(p, "TwoDimFunction")
    end
    
    @testset "TwoDimPath" begin
        x = collect(1:10) + rand(10)
        y = collect(1:10) + rand(10)
        path = TwoDimPath(x, y)
        
        p = plot(path)
    
        test_save(p, "TwoDimPath")
    end

    @testset "LabelledScatter" begin
        ls = LabelledScatter((x=rand(2, 100), y=rand(2, 100)))
        
        p = plot(ls)

        test_save(p, "LabelledScatter")
    end
    
    @testset "ImageGrid" begin
        ig = ImageGrid(rand(32, 32, 3, 100))
        
        p = plot(ig)
    
        test_save(p, "ImageGrid")
    end
    
    @testset "FeatureActivations" begin
        fa = FeatureActivations(rand(Bool, 20, 50))
        
        p = plot(fa)
    
        test_save(p, "FeatureActivations")
    end
end
