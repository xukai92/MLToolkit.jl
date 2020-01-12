using Test, LaTeXStrings, MLDatasets, Distributions
using MLToolkit.Plots

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
end

@testset "Plots" begin
    @testset "TwoYAxesLines" begin
        x = collect(1:0.5:10)
        y1 = sin.(x)
        y2 = x .^ 2
        
        p = TwoYAxesLines(x, y1, y2)
        fig = plot(p, "--o"; xlabel="x", ylabel1=L"\sin(x)", ylabel2=L"x^2")

        savefig(fig, p, joinpath(@__DIR__, "two_y_axes_lines.tex"))
        savefig(fig, p, joinpath(@__DIR__, "tetwo_y_axes_lines.png"); bbox_inches="tight")
    end

    @testset "ImageGrid" begin
        for (dataset, x) in [
            ("mnist", reshape(permutedims(MNIST.traintensor(Float32, 1:100), (2, 1, 3)), 784, :)),
            ("cifar10", permutedims(CIFAR10.traintensor(Float32, 1:100), (2, 1, 3, 4))),
        ]
            p = ImageGrid(x)
            fig = plot(p)

            savefig(fig, p, joinpath(@__DIR__, "imagegrid_$dataset.tex"))
            savefig(fig, p, joinpath(@__DIR__, "imagegrid_$dataset.png"); bbox_inches="tight")
        end
    end

    # # Two-dimensional density
    # p = plot(MvNormal(zeros(2), 1), (-3, 3), (-3, 3))

    # save("test_contour_data.tex", p, include_preamble=false)
    # save("test_contour_data.pdf", p)
end