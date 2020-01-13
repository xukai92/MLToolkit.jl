using Test, MLToolkit.Scripting

@testset "Scripting" begin
    @testset "Dict <-> NamedTuple" begin
        argdict = Dict(:a => 1, :b => "two", :c => true, :d => NaN)

        args = dict2namedtuple(argdict)
        @test keys(args) |> Set == (:a, :b, :c, :d) |> Set
        @test values(args) |> Set == (1, "two", true, NaN) |> Set
        @test args.a == 1
        @test args.b == "two"
        @test args.c == true
        @test isnan(args.d)

        argdict_copy = namedtuple2dict(args)
        @test argdict_copy[:a] == argdict[:a]
        @test argdict_copy[:b] == argdict[:b]
        @test argdict_copy[:c] == argdict[:c]
        @test isnan(argdict_copy[:d])
    end

    @testset "Addition for NamedTuple" begin
        t1 = (x=1, y=2)
        t2 = (x=3, y=4)
        tsum = t1 + t2
        @test tsum == (x=4, y=6)
        t = reduce(+, (t1, t2))
        @test t == tsum
    end

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

    include("args.jl")
    
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
