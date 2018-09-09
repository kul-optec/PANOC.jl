using Test
using PANOC

using Random

Random.seed!(0)

@testset "Iter tools" begin

    @testset "Looping" begin
        iter = rand(Float64, 10)
        last = PANOC.loop(iter)
        @test last == iter[end]
    end

    @testset "Halting" begin
        iter = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]
        last = PANOC.loop(PANOC.halt(iter, x -> x >= 1000))
        @test last == 1597
    end

    @testset "Side effects" begin
        # TODO
    end

    @testset "Sampling" begin
        iter = randn(Float64, 147)
        k = 0
        for x in PANOC.sample(iter, 10)
            idx = min(147, (k+1)*10)
            @test x == iter[idx]
            k = k+1
        end
    end

    @testset "Timing" begin
        iter = randn(Float64, 10)
        k = 0
        for (t, x) in PANOC.stopwatch(iter)
            @test x == iter[k+1]
            @test t >= k * 1e8 * 0.9 # 1e8 ns = 0.1 s, factor 0.9 is for safety
            sleep(0.1)
            k = k+1
        end
    end

end
