using LinearAlgebra
using Flux

function LecunUniform(fan_in::Integer, shape::Union{Integer,Tuple})::Array
    limit = √(3/fan_in)
    rand(shape...) * 2limit .- limit
end
function XavierNormal(fan_in::Integer, shape::Union{Integer,Tuple})::Array
    std = √(1/fan_in)
    randn(shape...) * std
end

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
    # eₘ = LecunUniform(d, d)
    eₘ = zeros(d)
    # eₕ = zeros(n)
    eₕ = LecunUniform(n, n)
    # eₓ = zeros(χ)
    eₓ = LecunUniform(χ, χ)

    # Wₕ = XavierNormal(n, (n,n))
    Wₕ = zeros(n,n)
    # Wₓ = XavierNormal(n, (n,χ))
    Wₓ = zeros(n,χ)
    Wₘ = XavierNormal(n, (n,d))

    A = zeros(d,d)
    B = zeros(d)

    for i in 0:(d-1)
        for j in 0:(d-1)
            _coef = i<j ? -1 : (-1)^(i-j+1)
            A[i+1,j+1] = (2i+1) * _coef
        end
        B[i+1] = (2i+1) * (-1)^i
    end
    A = A * (1/(θ-0.5)) #+ Matrix(I, d, d)
    B = B * (1/(θ-0.5))
    LMUCell(param(eₘ),param(eₓ),param(eₕ),param(Wₘ),param(Wₓ),param(Wₕ),A,B,activation,LMUCellState(zeros(n),zeros(d)))
end

function (c::LMUCell)(state::LMUCellState, x::Union{AbstractArray,Number})::Tuple{LMUCellState, AbstractArray}
    # x must be an array otherwise multiplication creates matrix instead of array
    x = x isa Number ? [x] : x
    h, m = state.h, state.m
    u = x⋅c.eₓ + h⋅c.eₕ + m⋅c.eₘ
    m = m + c.A*m + c.B*u
    h = c.activation.(
    c.Wₓ * x +
    c.Wₕ * h +
    c.Wₘ * m
    )
    @show h
    @show m
    return LMUCellState(Tracker.data(h),Tracker.data(m)), h
end

# function infer(m::Flux.Recur,xs...)
#     h, y = m.cell(m.state, xs...)
#     return y
# end
# function infer(m, x)
#     for layer in m.layers
#         if layer isa Flux.Recur
#             x = infer(layer, x)
#         else
#             x = layer(x)
#         end
#     end
#     out = m(x)[end]
#     Flux.reset!(m)
#     out
# end

# Flux.params(c::LMUCell)::Flux.Params = Flux.Params([c.Wₕ, c.Wₓ,c.Wₘ, c.eₓ, c.eₕ, c.eₘ])

Flux.hidden(c::LMUCell) = c.state

Flux.@treelike LMUCell


m = Chain( Flux.LSTM(1,100),
 Dense(100,5) )

m.layers[1].cell.Wₕ
m.layers[1].cell.Wₓ
m.layers[1].cell.eₓ
m.layers[1].cell.eₕ
m.layers[1].cell.eₘ
m.layers[1].cell.A

interval = collect(0:0.0001:20π)
# train_x = sin.(interval) .+ randn(length(interval))*0.05
train_x = randn(length(interval))
window = 100

function loss(model, x, y)
    _eval = model(x)
    # @show _eval, y, length(y)
    # _l = sum((_eval .- y).^2) * 1 // length(y)
    # return _l
    Flux.mse(_eval,y)
end

function seek_backward(data, index)
    global window
    data[index-window,:]
end


# m([0.5])
Flux.reset!(m)
[m([x]) for x in 0.01:0.001:1.0]

opt = ADAM(0.1)
Flux.reset!(m)
ps = Flux.params(m)
# ps = Flux.params(m.layers[1].cell.Wₘ)
for i in 1:1
    global train_x
    # train_recurrent!(m, train_x, ps, opt, window, seek_backward)
    train_memory!(m,train_x,ps,opt,ceil(Integer,θ),θ,k)
    Flux.reset!(m)
    train_x = randn(length(interval))
    # if i % 5 == 0
        # println("Epoch: $i")
        # println("   Loss: $(loss(batch[1][1], batch[1][2]).data)")
    # end
end


m.layers[1].cell.Wₘ


m.layers[1].cell.Wₘ



indeces = [floor(Integer, i*T / (k-1) ) for i in 0:(k-1)]

for i in 1:ceil(Integer,θ)
    m(train_x[i,:])
end
_l = 0.

# for i in (ceil(Integer,θ)+1):size(train_x,1)
i = i+1
i = ceil(Integer,θ)+1
dₜ = train_x[i,:]
y = train_x[i.-indeces]
_eval = m(dₜ)

gs = Tracker.gradient(() -> Flux.mse(_eval,y), ps)
sum(first(gs.grads)[1].grad .!= 0)
Flux.Tracker.update!(opt, ps, gs)
_l = 0.
        # Flux.truncate!(model)
# end

#region draw
using Plots
scatter(interval, train_x)

function plot_sine()
    ŷ = []
    y = []
    Flux.reset!(m)
    for i in 1:1000
        tmp = Tracker.data(m(train_x[i,:]))
        push!(ŷ, train_x[i])
        push!(y, train_x[i])
    end
    for i in 1001:5000
        last = train_x[i,:]
        # last = ŷ[end,:]
        push!(ŷ, Tracker.data(m(last))[1])
        push!(y, train_x[i])
    end

    # scatter!(y)
    scatter(ŷ)
end
#endregion

"""
Training for recurrent neural networks, using cold start (not training initially)
    Expects data to be in the form (timesteps x features)

    lag𝕗 = lag𝕗(data, current index) should return the point trying to predict
"""
function train_recurrent!(model, data, ps, opt, cold_start, lag𝕗)
    indeces = []
    for i in 1:cold_start
        model(data[i,:])
    end
    _l = 0.
    for i in (cold_start+1):size(data,1)
        dₜ = data[i,:]
        try
            # lag = lag𝕗(data,i)
            # if any(isnan.(lag)) break end

            # _l += loss(model, dₜ, lag)
            y =
            _l = loss(model, dₜ, )
            if i % 10 == 0

                if i%3000 == 0
                    println("Step: $(i-cold_start)   Loss: $_l")
                end

                gs = Tracker.gradient(() -> _l, ps)
                Flux.Tracker.update!(opt, ps, gs)
                _l = 0.

                # if i%100 == 0
                #     tmp_state = LMUCellState(m.layers[1].state.h, m.layers[1].state.m)
                #     plt = plot_sine()
                #     display(plt)
                #     m.layers[1].state = tmp_state
                # end
            end
        catch ex
            if ex isa InterruptException
                return
            else
                rethrow(ex)
            end
        end
    end
end


function train_memory!(model, data, ps, opt, cold_start, T, k)

    indeces = [floor(Integer, i*T / (k-1) ) for i in 0:(k-1)]

    for i in 1:cold_start
        model(data[i,:])
    end
    _l = 0.
    for i in (cold_start+1):size(data,1)
        dₜ = data[i,:]
        try
            y = data[i.-indeces]
            _l = loss(model, dₜ, y)
            # if i % 20 == 0

            if i%3000 == 0
                println("Step: $(i-cold_start)   Loss: $(Tracker.data(_l))")
            end

            gs = Tracker.gradient(() -> _l, ps)
            Flux.Tracker.update!(opt, ps, gs)
            _l = 0.
            # end
            Flux.truncate!(model)
        catch ex
            if ex isa InterruptException
                return
            else
                rethrow(ex)
            end
        end
    end
end


plt = plot_sine()
display(plt)
