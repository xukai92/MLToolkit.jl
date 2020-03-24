using Test, MLToolkit.Scripting
using ArgParse: ArgParseSettings, @add_arg_table

@testset "Scripting" begin
    @testset "Addition for NamedTuple" begin
        t1 = (x=1, y=2)
        t2 = (x=3, y=4)
        tsum = t1 + t2
        @test tsum == (x=4, y=6)
        t = reduce(+, (t1, t2))
        @test t == tsum
    end

    @testset "find_latestdir" begin
        @test DATETIME_FMT == "ddmmyyyy-H-M-S"
        @test find_latestdir(@__DIR__) == "20051994-12-34-56"
    end

    @testset "parse_toml" begin
        tomlpath = joinpath(@__DIR__, "Test.toml")

        argdict = parse_toml(tomlpath, (:level1 => "a1", :level2 => "b1",))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 1
        @test argdict[:f3] == 1

        argdict = parse_toml(tomlpath, (:level1 => "a1", :level2 => "b2",))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 1
        @test argdict[:f3] == 2

        @test_throws AssertionError parse_toml(tomlpath, (:level1 => "a2", :level2 => "b1",))

        argdict = parse_toml(tomlpath, (:level1 => "a2", :level2 => "b1", :level3 => "c1"))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 2
        @test argdict[:f3] == 3
        @test argdict[:f4] == 1

        argdict = parse_toml(tomlpath, (:level1 => "a2", :level2 => "b2",))
        @test argdict[:f1] == 1
        @test argdict[:f2] == 2
        @test argdict[:f3] == 4
    end

    @testset "parse_argstr" begin
        s = ArgParseSettings()

        @add_arg_table! s begin
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
    
    @testset "Scripting.check" begin
        x = 1
        
        @test @jupyter(:(x + 1), default=3) == 2
        @test @script(:(x + 1), default=3) == 3
        @test @tb(:(x + 1), default=3) == 2
        @test @wb(:(x + 1), default=3) == 2
        
        vc = [1 Inf 3; NaN 5 6]
        vm1 = [1 2 3; 3 4 5]
        vm2 = [11 22 33; 44 55 66]

        check_strs = Scripting.checknumerics_strs(vc, vm1, vm2; vcheckname="vc", vmonitornames=["vm1", "vm2"])
        @test check_strs[1] == "vc[2,1] = NaN\n  vm1[2,1] = 3\n  vm2[2,1] = 44"
        @test check_strs[2] == "vc[1,2] = Inf\n  vm1[1,2] = 2\n  vm2[1,2] = 22"

        @warn "`@checknumerics` is not tested."
    end
end
