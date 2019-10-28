using Test, MLToolkit
using ArgParse: ArgParseSettings, @add_arg_table

@testset "Scripting.args" begin
    FILEDIR = first(splitdir(@__FILE__))

    @testset "argstring" begin
        argdict = Dict(:a => 1, :b => "two", :d => NaN, :c => true)
        @test argstring(argdict) == "--a 1 --b two --d NaN --c true"
    end

    @testset "argstring_flat" begin
        args = Dict(:a => 1, :b => "two", :c => true, :d => NaN)
        for delimiter in ["-", ","],
            eqsym in ["-", "="]
            argstr_flat = argstring_flat(args; exclude=[:d], delimiter=delimiter, eqsym=eqsym)
            @test argstr_flat == "a$(eqsym)1$(delimiter)b$(eqsym)two$(delimiter)c$(eqsym)true"
        end
        argstr_flat = argstring_flat(args; include=[:a])
        @test argstr_flat == "a=1"
    end

    @testset "find_latestdir" begin
        @test DATETIME_FMT == "ddmmyyyy-H-M-S"
        @test find_latestdir(FILEDIR) == "20051994-12-34-56"
    end

    @testset "parse_toml" begin
        tomlpath = FILEDIR * "/Test.toml"

        argdict = parse_toml(tomlpath, ((:level1 => "a1"), (:level2 => "b1"),))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 1
        @test argdict[:f3] == 1

        argdict = parse_toml(tomlpath, ((:level1 => "a1"), (:level2 => "b2"),))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 1
        @test argdict[:f3] == 2

        @test_throws AssertionError parse_toml(tomlpath, ((:level1 => "a2"), (:level2 => "b1"),))

        argdict = parse_toml(tomlpath, ((:level1 => "a2"), (:level2 => "b1"), (:level3 => "c1")))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 2
        @test argdict[:f3] == 3
        @test argdict[:f4] == 1

        argdict = parse_toml(tomlpath, ((:level1 => "a2"), (:level2 => "b2"),))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 2
        @test argdict[:f3] == 4
    end

    @testset "parse_argstr" begin
        s = ArgParseSettings()

        @add_arg_table s begin
            "--a"
                arg_type = Int64
                required = true
            "--b"
                arg_type = Float64
                required = true
            "--c"
                arg_type = String
                default = nothing
        end

        argstr = "--a 1 --b 2.0"
        argdict = parse_argstr(argstr, s; as_symbols=true)
        @test argdict[:a] == 1
        @test argdict[:b] == 2.0
        @test argdict[:c] == nothing

        argdict = parse_argstr(argstr * " --c c", s; as_symbols=true)
        @test argdict[:c] == "c"
    end
end
