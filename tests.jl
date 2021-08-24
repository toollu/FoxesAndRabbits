using FoxesAndRabbits
using Test


## unit tests

test_int = 1
test_float = 5.0
test_vector = [1.0, 2.0, 3.0, 4.0]

@testset "make artificial data" begin
    @test make_population_data([test_float, test_float], test_float, stochastic = true) isa PopulationData
    @test make_population_data([test_float, test_float], test_float, stochastic = false) isa PopulationData
    @test make_population_data([test_float, test_float], test_float, stochastic = false, parameters = test_vector) isa PopulationData
    @test_throws AssertionError make_population_data([test_float, test_float], 1.0, stochastic = true)
    @test_throws TypeError  make_population_data([test_float, test_float], test_float, stochastic = 1)
    @test_throws MethodError  make_population_data([test_float, test_float], test_float, 1)
    @test_throws MethodError make_population_data([test_int, test_int], test_int, stochastic = false)
end

#other units

## integration tests
