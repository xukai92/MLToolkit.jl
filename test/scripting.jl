using Test, MLToolkit
using ArgParse: ArgParseSettings, @add_arg_table

@testset "Scripting" begin
    @warn "`find_latest_dir()` is not tested."

    @testset "parsetoml" begin
        tomlpath = first(splitdir(@__FILE__)) * "/Test.toml"
        argdict = parsetoml(tomlpath, ((:level1 => "a1"), (:level2 => "b1"),))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 1
        @test argdict[:f3] == 1
        argdict = parsetoml(tomlpath, ((:level1 => "a1"), (:level2 => "b2"),))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 1
        @test argdict[:f3] == 2
        @test_throws AssertionError parsetoml(tomlpath, ((:level1 => "a2"), (:level2 => "b1"),))
        argdict = parsetoml(tomlpath, ((:level1 => "a2"), (:level2 => "b1"), (:level3 => "c1")))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 2
        @test argdict[:f3] == 3
        @test argdict[:f4] == 1
        argdict = parsetoml(tomlpath, ((:level1 => "a2"), (:level2 => "b2"),))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 2
        @test argdict[:f3] == 4
    end

    @testset "parse_args" begin
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

        args_str = "--a 1 --b 2.0"
        args = parse_args(args_str, s; as_symbols=true)
        @test args[:a] == 1
        @test args[:b] == 2.0
        @test args[:c] == nothing

        args = parse_args(args_str * " --c c", s; as_symbols=true)
        @test args[:c] == "c"
    end

    @testset "flatten_dict" begin
        args = Dict(:a => 1, :b => "two", :c => true, :d => NaN)
        for delimiter in ["-", ","],
            equal_sym in ["-", "="]
            flat_args = flatten_dict(args; exclude=[:d], delimiter=delimiter, equal_sym=equal_sym)
            @test flat_args == "a$(equal_sym)1$(delimiter)b$(equal_sym)two$(delimiter)c$(equal_sym)true"
        end
        flat_args = flatten_dict(args; include=[:a])
        @test flat_args == "a=1"
    end

    @testset "dict2namedtuple" begin
        args_dict = Dict(:a => 1, :b => "two", :c => true, :d => NaN)
        args = dict2namedtuple(args_dict)
        @test keys(args) |> Set == (:a, :b, :c, :d) |> Set
        @test values(args) |> Set == (1, "two", true, NaN) |> Set
        @test args.a == 1
        @test args.b == "two"
        @test args.c == true
        @test isnan(args.d)
    end

    @testset "merge_namedtuples" begin
        t1 = (x=1, y=2)
        t2 = (x=3, y=4)
        t = merge_namedtuples(sum, t1, t2)
        @test t == (x=4, y=6)
    end

    @testset "args_dict2str" begin
        args_dict = Dict(:a => 1, :b => "two", :d => NaN, :c => true)
        @test args_dict2str(args_dict) == "--a 1 --b two --d NaN --c true"
    end

    @warn "`jupyter()` is not tested."
    @warn "`@jupyter` is not tested."
    @warn "`checknumerics()` is not tested."
    @warn "`@checknumerics` is not tested."

    @testset "sweepcmd" begin
        @test sweepcmd("sleep @Ts @D", "@T" => [1, 2], "@D" => [3, 4]) == [`sleep 1s 3`, `sleep 2s 3`, `sleep 1s 4`, `sleep 2s 4`]
        @test sweepcmd("sleep @Ts @D", :T => [1, 2], :D => [3, 4]) == [`sleep 1s 3`, `sleep 2s 3`, `sleep 1s 4`, `sleep 2s 4`]
    end

    @testset "sweeprun" begin
        # Check if runs are in parallel
        t = @elapsed sweeprun("sleep @Ts", "@T" => [1, 2, 3, 4])
        @test t < 5
        # Check if runs are not in parallel
        t = @elapsed sweeprun("sleep @Ts", "@T" => ones(4); maxasync=1)
        @test t > 1
    end

    @warn "`figure_to_image` is not tested."
end
