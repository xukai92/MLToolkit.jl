using Test, MLToolkit
using ArgParse: @add_arg_table, ArgParseSettings

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
end
