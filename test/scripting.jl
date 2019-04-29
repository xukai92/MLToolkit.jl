using Test, MLToolkit
using ArgParse: ArgParseSettings, @add_arg_table

@testset "Scripting" begin
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

    @warn "`jupyter()` is not tested."
    @warn "`@jupyter` is not tested."
    @warn "`checknumerics()` is not tested."
    @warn "`@checknumerics` is not tested."

    @testset "sweepcmd" begin
        @test sweepcmd("sleep @Ts", Dict("@T" => [1, 2, 3])) == [`sleep 1s`, `sleep 2s`, `sleep 3s`]
    end

    @testset "sweeprun" begin
        # Check if runs are in parallel
        t = @elapsed sweeprun("sleep @Ts", Dict("@T" => [1, 2, 3, 4]))
        @test t < 5
    end
    
end
