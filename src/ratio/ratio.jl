using Distances: pairwise, SqEuclidean
using LinearAlgebra: diagm
using Statistics: median
import JuMP, Ipopt
include("moment_matching.jl")
export estimate_r_de, get_r_hat_numerically, get_r_hat_analytical