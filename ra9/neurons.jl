using Calculas
using CUDA 

# Sigmoid Function σ(x) = 1/1+ℯ^(-x)

function sigmoid(input::Real)
    return 1.0 / (1.0 + exp(-input))
end

#Softmax Function

function softmax(input::Real)
end

#Spike Response Model 

function SRM(input::Real)
end

#Integrated and Fire Model

function IF(input::Real,tau_ref,min_volatage,amplitude,initial_state)
end

#Leaky Integrated and Fire Model

function LIF(input::Real,tau_rc,tau_ref,min_volatage,amplitude,initial_state)
end
