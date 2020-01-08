using LinearAlgebra
using Flux
using Polynomials

include("utils.jl")

struct LMUCellState{V}
    h::V   # Hidden state
    m::V   # Memory state
end

mutable struct LMUCell{M,Mₜ,V,Vₜ}
    eₘ::Vₜ  # subscript ₜ stands for tracked (A,B are not)
    eₓ::Vₜ
    eₕ::Vₜ
    Wₘ::Mₜ
    Wₓ::Mₜ
    Wₕ::Mₜ
    A::M
    B::V
    activation::Function
    state::LMUCellState
end

"""
Initialises LMUCell.
d:  memory size
n:  hidden size
χ:  input size
Δt: time between events
θ:  window length
activation: non-linear activation function
"""
function LMUCell(d,n,χ,Δt,θ, activation=tanh)::LMUCell
    eₘ = LecunUniform(d, d)
    eₕ = LecunUniform(n, n)
    eₓ = LecunUniform(χ, χ)

    Wₕ = XavierNormal(n, (n,n))
    Wₓ = XavierNormal(n, (n,χ))
    Wₘ = collect( transpose(LegendreInitialiser(d,n) ) )

    A,B = LegendreDelay(d, θ-0.5)
    A,B = ZOH(A,B,d, 1.)

    LMUCell(eₘ,eₓ,eₕ,Wₘ,Wₓ,Wₕ,A,B,activation,LMUCellState(zeros(n),zeros(d)))
end

function (c::LMUCell)(state::LMUCellState, x::Union{AbstractArray,Number})::Tuple{LMUCellState, AbstractArray}
    # x must be an array otherwise multiplication creates matrix instead of array
    x = x isa Number ? [x] : x
    h, m = state.h, state.m
    u = x⋅c.eₓ + h⋅c.eₕ + m⋅c.eₘ
    m = m + c.A'*m + c.B*u
    h = c.Wₓ * x +
    c.Wₕ * h +
    c.Wₘ * m

    return LMUCellState(h,m), h
end

""" Infer next state of RNN without modifying it """
function infer(m::Flux.Recur,xs...)
    h, y = m.cell(m.state, xs...)
    return y
end

""" Infer next state of RNN without modifying it """
function infer(m, x)
    for layer in m.layers
        if layer isa Flux.Recur
            x = infer(layer, x)
        else
            x = layer(x)
        end
    end
    out = m(x)[end]
    Flux.reset!(m)
    out
end

Flux.hidden(c::LMUCell) = c.state

Flux.@functor LMUCell
