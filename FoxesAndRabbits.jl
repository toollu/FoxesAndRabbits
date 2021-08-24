module FoxesAndRabbits
#author: Tobias Seidler
#used julia code style convention: blue

#load external libraries
using DifferentialEquations
using DiffEqFlux
using Distributions
using GalacticOptim
using Plots

#expose module functionalities
export make_population_data
export optimize!
export predict
export visualize

## constants
const population_pars = [1.5, 1.0, 3.0, 1.0] #default model parameters

## ODE System definitions
function lotka_volterra!(du,u,p,t) #standard lotka volterra model
    a, b, c, d = p #unpack growth/death parameters
    du[1] = u[1]*(a-b*u[2])
    du[2] = -u[2]*(c-d*u[1])
end

function σ_volterra!(du,u,p,t) #stochastic ODE noise/std
    du[1] = 0.15
    du[2] = 0.12
end

neural_RHS = FastChain(FastDense(2, 20, tanh),FastDense(20, 1)) #simple shallow tanh layer
function neural_lotka_volterra!(du,u,θ,t)#neural ode model
    du[1] = neural_RHS(u,θ)[1]
    du[2] = -u[2]*(population_pars[3]-population_pars[4]*u[1])
end

## data/model types,structs & artificial data initialization
abstract type Population end

struct PopulationData{T<:Real} <: Population #generic data holding structure
    time::Vector{T}
    foxes::Vector{T}
    rabbits::Vector{T}
end

struct PredictedPopulationData{T} <: Population #composite type for predictions
    trueData::PopulationData
    predictedData::PopulationData
    parameters::Vector{T}
end

function make_population_data(
        initial_population::Vector{T},
        tmax::T;
        ΔT::T = 0.1, #default for timestep
        stochastic::Bool = false, #introduce some noise by default
        parameters::Vector{T} =  population_pars, #take default constants
    ) where T <: Real
        #make  time variables for ODE
        @assert ΔT<tmax/10 #check that we at least have 10 time points
        tspan = (0.0,tmax)
        save_times = 0:ΔT:tmax
        #generate (stochastic) data of foxes & rabbits
        if stochastic == true
                prob = SDEProblem(lotka_volterra!,σ_volterra!,initial_population,tspan,parameters)
        else
                prob = ODEProblem(lotka_volterra!,initial_population,tspan,parameters)
        end
        #shape data & make struct
        sol = Array(solve(prob,saveat = save_times))
        return PopulationData{typeof(tmax)}(save_times,sol[2,:],sol[1,:])
end

## training, prediction & optimiziation functions
function initialize_problem(θ,data) #function to initialize neuralODE
    u0 = [data.rabbits[1], data.foxes[1]]
    tspan = (data.time[1],data.time[end])
    return ODEProblem(neural_lotka_volterra!,u0,tspan,θ)
end

function predict(θ,data) #function to initialize & predict model given parameters & data
    prob = initialize_problem(θ,data)
    sol = solve(prob,KenCarp4(),saveat = data.time, reltol=1e-6, sensealg = ForwardDiffSensitivity())
    return Array(sol)[1,:]
end

function predict_ensemble(data,θ,uncertainty; trajectories = 50) #initialize model & run prediction ensemble
    #problem modification function
    ensemble_variation(prob,i,repeat) = remake(prob, u0 = rand(Uniform(1-uncertainty,1+uncertainty))*prob.u0)
    #set up ensemble simulation and run
    prob = initialize_problem(θ,data)
    ensemble_prob = EnsembleProblem(prob, prob_func = ensemble_variation)
    ensemble_sol = solve(ensemble_prob, KenCarp4(), EnsembleThreads(), trajectories=trajectories)
    return ensemble_sol
end

callback = function(p,l) #callback function for optimizer
  println(l)             #prints loss and prematurely exits training if loss sufficiently small
  if l < 1e-3
      return true
  else
      return false
  end
end

function loss(θ,data) #calculates RMSE based on model prediction
    rabbits_mod = predict(θ,data)
    return sum(abs2, rabbits_mod .- data.rabbits)
end

function learn!(data::PopulationData; θ = nothing, iterations = 500) #learn neural ode  based on given data
    #intialize random parameters for NN if  θ not given
    θ = (θ == nothing ? Float64.(initial_params(neural_RHS)) : θ)
    #build optimization function with autodiff& problem
    optfun = OptimizationFunction((p,x) -> loss(p,data), GalacticOptim.AutoZygote())
    prob = GalacticOptim.OptimizationProblem(optfun, θ)
    #optimize and return  resulting (new) population data & optimized parameters
    res = GalacticOptim.solve(prob,ADAM(0.1),cb = callback, maxiters = iterations)
    θ = res.minimizer
    rabbits = predict(θ,data)
    modelled_data = PopulationData(data.time,data.foxes,rabbits)
    return PredictedPopulationData(data,modelled_data,θ)
end

## Data Visualization: dispatch on different types of data
function visualize(data::PopulationData)
    plot(data.time,data.foxes, label = "Foxes")
    plot!(data.time,data.rabbits, label = "Rabbits")
    xlabel!("Time"); ylabel!("N")
end

function visualize(data::PredictedPopulationData)
    tru = data.trueData
    pre = data.predictedData
    plot(pre.time,pre.rabbits, label = "Rabbits: Model")
    scatter!(tru.time,tru.rabbits, color = 1, label = "Rabbits: Data")
    plot!(tru.time, tru.foxes, color = 2, label = "Foxes")
    xlabel!("Time"); ylabel!("N")
end

function visualize(data::EnsembleSolution)
    summ = EnsembleSummary(data,0:0.1:5)
    plot(summ,fillalpha=0.5,label=["Rabbits" "Foxes"],legend = true, title = "95% quantiles")
    xlabel!("Time"); ylabel!("N")
end

end #module FoxesAndRabbits


# ##Script, executed as in documentation
# #(Results will be different due to stochastic data generation)
#
# using FoxesAndRabbits
#
# #make population and fit hybrid model
# population = make_population_data([1.0, 1.0], 8.0, stochastic = true)
# pred_pop = learn!(population)
#
# #make test set, continuing in time
# initial_rabbits = population.rabbits[end]
# initial_foxes = population.foxes[end]
# t_end = 5.0
# new_population = make_population_data([initial_rabbits, initial_foxes], t_end, stochastic = true)
#
# #predict and make composed data struct
# predicted_rabbits = predict(pred_pop.parameters,new_population)
# modelled_data = PopulationData(new_population.time,new_population.foxes,predicted_rabbits)
# test_population = PredictedPopulationData(new_population,modelled_data,[])
#
# #run ensemble
# uncertainty = 0.3
# trajectories = 50
# ensemble_sol = predict_ensemble(new_population,pred_pop.parameters, uncertainty, trajectories = trajectories)
#
# #plot everything
#
# visualize(population)
# visualize(pred_pop)
# visualize(test_population)
# visualize(ensemble_sol)
