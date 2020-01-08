polydot(p::Poly, q::Poly) = polyint(p*q, -1,1)

function LegendrePolynomials(n)
    legendre = [Poly([1.])]
    for i = 1:n
        p = Poly([k == i ? 1. : 0. for k=0:i])
        for q in legendre
            p = p - q * (polydot(q, p) / polydot(q,q))
        end
        push!(legendre, p / p(1.))
    end
    return legendre
end

function LegendreInitialiser(rows, cols)
    linspace = collect(range(-1,stop=1,length=cols))
    pols = LegendrePolynomials(rows)

    result = zeros(rows,cols)

    for i in 1:rows
        result[i,:] = pols[i].(linspace)
    end
    result
end


function LecunUniform(fan_in::Integer, shape::Union{Integer,Tuple})::Array
    limit = √(3/fan_in)
    rand(shape...) * 2limit .- limit
end
function XavierNormal(fan_in::Integer, shape::Union{Integer,Tuple})::Array
    std = √(1/fan_in)
    randn(shape...) * std
end

function LegendreDelay(order, θ)
    A = zeros(order,order)
    B = zeros(order)

    for i in 0:(order-1)
        for j in 0:(order-1)
            _coef = i<j ? -1 : (-1)^(i-j+1)
            A[i+1,j+1] = (2i+1) * _coef
        end
        B[i+1] = (2i+1) * (-1)^i
    end
    A = A * (1.0/(θ))
    B = B * (1.0/(θ))
    A,B
end

""" Zero-hold Discretization """
function ZOH(A::AbstractArray, B::AbstractArray, order::Integer, dt::AbstractFloat)
    nx = order
    nu = 1
    M = exp([A*dt  B*dt; zeros(nu, nx + nu)])
    Ad = M[1:nx, 1:nx] - Matrix(I, nx, nx)
    Bd = M[1:nx, nx+1:nx+nu]
    Ad,Bd
end
