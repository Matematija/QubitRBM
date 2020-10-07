using PastaQ
using ITensors
using LightGraphs
# using GraphIO
using DelimitedFiles
using NPZ

import PastaQ: gate

angles_path = length(ARGS) > 1 ? ARGS[2] : "./angles_p2_N20.txt"
edgelist_path = length(ARGS) > 2 ? ARGS[3] : "./edgelist_p2_N20.txt" 

angles = readdlm(angles_path, ',', Float64)[:,1];
# g = SimpleGraph(loadgraph(edgelist_path, "graph_key", EdgeListFormat()))

####################################################################################

# N = length(vertices(g)); # qubits
N = parse(Int, ARGS[1])
p = length(angles)÷2;
k = 3;

g = random_regular_graph(N, k)

NDIMS = parse(Int, ENV["SLURM_ARRAY_TASK_COUNT"])
PROC = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
MAXDIM = 10000;
MINDIM = 50;
DIM = MINDIM + PROC*((MAXDIM-MINDIM)÷NDIMS)
# DIMS = [floor(Int, d) for d in range(MINDIM, MAXDIM, length=NDIMS)]
# DIM = DIMS[PROC]

NSAMPLES = 20000

edgelist = [(x.src,x.dst) for x in edges(g)];

γs = angles[1:p];
βs = angles[(p+1):end];

println("Parameters set:")
println("N: ", N)
println("p: ", p)
println("k: ", k)
println("gammas: ", γs)
println("betas: ", βs)
println("PROC: ", PROC)
println("DIM: ", DIM)

flush(stdout)

####################################################################################

function cost(samples::Array{T,2}, edgelist::Array{Tuple{Int64,Int64},1}) where {T <: Number}
    
    res = zeros(size(samples, 1))
    z = (-1) .^samples
    
    for (i,j) in edgelist
        res += prod(z[:,[i,j]], dims=2)
    end
    
    return res
end

function mean(arr::Array{T,N}) where {T,N}
    return sum(arr) ./length(arr)
end
    
function mean(arr::Array{T,N}, dims::Int64) where {T,N}
    return sum(arr, dims=dims) ./size(arr, dims)
end

println("Defining the circuit...")
flush(stdout)

gate(::GateName"RZZ"; ϕ::Number) =
  [     1           0          0         0
        0      exp(2*im*ϕ)     0         0
        0           0     exp(2*im*ϕ)    0
        0           0          0         1      ];

Hs = [("H", n) for n in 1:N];

UC(edgelist::Array{Tuple{Int64,Int64},1}, γ::Float64) = [("RZZ", (i, j), (ϕ=γ,)) for (i,j) in edgelist]
UB(N::Int64, β::Float64) = [("Rx", i, (θ=2*β,)) for i in 1:N]

gates = vcat(Hs, UC(edgelist, γs[1]), UB(N, βs[1]));

for layer in 2:p
    append!(gates, UC(edgelist, γs[layer]))
    append!(gates, UB(N, βs[layer]))
end

println("Calculating the MPS state...")
flush(stdout)
@time ψ = runcircuit(N, gates, maxdim=DIM)

println("Sampling the MPS state...")
flush(stdout)
@time samples = getsamples(ψ, NSAMPLES);

normalize!(ψ)

costs = cost(samples, edgelist);

filename = "data_"*string(PROC)*".npz"

println("Saving the output file to ", filename)
flush(stdout)
npzwrite(filename, Dict("samples" => samples,
                        "costs" => costs,
                        "mps_dim" => DIM,
                        "gammas" => γs,
                        "betas" => βs,
                        "edgelist" => hcat([[u,v] for (u,v) in edgelist]...)))

println("Mean cost: ", mean(costs))
flush(stdout)