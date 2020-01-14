using Test, MLToolkit.Plots, LaTeXStrings

@testset "Utilites" begin
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
end

@testset "Plots" begin
    function test_savefig(fig, p, fname; tex=true)
        savefig(fig, p, joinpath(@__DIR__, "$fname.png"))
        tex && savefig(fig, p, joinpath(@__DIR__, "$fname.tex"))
    end
    
    @testset "TwoYAxesLines" begin
        x = collect(1:0.5:10)
        y1 = sin.(x)
        y2 = x .^ 2
        
        p = TwoYAxesLines(x, y1, y2)
        fig = plot(p, "--o"; xlabel="x", ylabel1=L"\sin(x)", ylabel2=L"x^2")

        test_savefig(fig, p, "TwoYAxesLines")
    end

    @testset "OneDimFunction" begin
        using Distributions

        p = Plot(Normal(0, 1))
        fig = plot(p, (-3, 3); figsize=(5, 5))

        test_savefig(fig, p, "OneDimFunction"; tex=false)   # FIXME: cannot save this to .tex file
    end

    @testset "ImageGrid" begin
        using MLDatasets

        for (dataset, x) in [
            ("mnist", reshape(permutedims(MNIST.traintensor(Float32, 1:100), (2, 1, 3)), 784, :)),
            ("cifar10", permutedims(CIFAR10.traintensor(Float32, 1:100), (2, 1, 3, 4))),
        ]
            p = ImageGrid(x)
            fig = plot(p)

            test_savefig(fig, p, "ImageGrid_$dataset")
        end
    end

    @testset "TwoDimFunction" begin
        using Distributions

        p = Plot(MvNormal(zeros(2), 1))
        fig = plot(p, (-3, 3), (-3, 3); figsize=(5, 5))

        test_savefig(fig, p, "TwoDimFunction")
    end

    @testset "FeatureActivations" begin
        Z = rand(Bool, 20, 50)

        p = FeatureActivations(Z)
        fig = plot(p)
        
        test_savefig(fig, p, "FeatureActivations")
    end
end
