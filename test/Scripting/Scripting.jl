using Test, MLToolkit

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

    @testset "Reduce NamedTuple" begin
        t1 = (x=1, y=2)
        t2 = (x=3, y=4)
        t = reduce(sum, (t1, t2))
        @test t == (x=4, y=6)
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
    
    include("check.jl")
end
