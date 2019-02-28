"""
Bouligand-Levenberg-Marquardt iteration for the nonsmooth inverse problem 
F(u) = y^δ with y = F(u) solving the nonsmooth semilinear elliptic equation

(1)     -Δy + max(y,0) = u in Ω, y = 0 on ∂Ω.

The Bouligand-Levenberg-Marquardt method is defined as

(2)     u^δ\\_{n+1} = u^δ\\_{n} + 
            (G\\_{u^δ\\_n}*G\\_{u^δ\\_n} + α\\_n I)^{-1}(y^δ - F(u^δ\\_n))

for G\\_{u^δ\\_n} a Bouligand subderivative of S at u^δ\\_n. The iteration
is stopped with the disrepancy principle. F is evaluated by solving (1) using
a semismooth Newton method.  
For details, see

Christian Clason, Vu Huu Nhu:
Bouligand-Levenberg-Marquardt iteration for a non-smooth ill-posed problem,
arXiv:1902.10596
"""
module BouligandLevenbergMarquardt

using LinearAlgebra,SparseArrays
using Printf

export run_example,FEM

"finite element structure holding mesh and assembled matrices"
struct FEM
    N::Int64                         # number of vertices per dimension
    x::Vector{Float64}               # x coordinates of inner nodes
    y::Vector{Float64}               # y coordinates of inner nodes
    A::SparseMatrixCSC{Float64}      # stiffness matrix
    AT::SparseMatrixCSC{Float64}     # adjoint stiffness matrix
    M::SparseMatrixCSC{Float64}      # mass matrix
    LM::SparseMatrixCSC{Float64}     # lumped mass matrix
    ndof::Int64                      # number of degrees of freedom
    norm                             # function computing the L2 norm
    relerror                         # function computing the relative error
end

"generate grid and assemble matrices for N×N uniform grid"
function FEM(N::Int64)
    # set up mesh
    dx = 1/N
    xm = range(dx,step=dx,length=N-1)
    xx,yy = reshape(xm, 1, N-1), reshape(xm, N-1, 1)
    x,y = repeat(xx,outer=(N-1,1))[:], repeat(yy,outer=(1,N-1))[:]
    ndof = (N-1)*(N-1)
    # set up stiffness matrix, lumped mass matrix
    e  = fill(1.0,N-1)
    D2 = spdiagm(-1=>-e[2:end],0=>2e,1=>-e[2:end])
    Id = sparse(I,N-1,N-1)
    A  = kron(D2,Id)+kron(Id,D2)
    AT = SparseMatrixCSC(A') # precompute, otherwise addition slow
    LM = sparse(dx*dx*I,ndof,ndof)
    # set up mass matrix
    rows = Int64[]; cols = Int64[]; vals = Float64[]
    for i=1:N-1
        for j=1:N-1
            # entries on diagonal
            ind_node = (N-1)*(j-1) + i-1
            append!(vals,0.5*dx*dx)
            append!(rows,ind_node)
            append!(cols,ind_node)
            # entries off diagonal
            val = 1/12*dx*dx
            if i < N-1     # right vertex
                append!(vals,val)
                append!(rows,ind_node+1)
                append!(cols,ind_node)
                append!(vals,val)
                append!(rows,ind_node)
                append!(cols,ind_node+1)
            end
            if j < N-1     # top vertex
                append!(vals,val)
                append!(rows,ind_node+(N-1))
                append!(cols,ind_node)
                append!(vals,val)
                append!(rows,ind_node)
                append!(cols,ind_node+N-1)
            end
            if (i<N-1) & (j<N-1) # top right vertex
                append!(vals,val)
                append!(rows,ind_node+N)
                append!(cols,ind_node)
                append!(vals,val)
                append!(rows,ind_node)
                append!(cols,ind_node+N)
            end
        end
    end
    M = sparse(rows.+1,cols.+1,vals)
    # functions to compute L2 norms and relative errors
    l2norm = (u)->sqrt(u'*M*u)
    relerror = (u,v)->l2norm(u-v)/l2norm(v)
    FEM(N,x,y,A,AT,M,LM,ndof,l2norm,relerror)
end

"evaluate forward mapping by solving (1) using semismooth Newton method" 
function F(fem::FEM,u,yn=zero(u))
    En = yn.>=0; Enew = similar(En)
    converged = false
    rhs = similar(yn)
    while !converged
        DN = spdiagm(0=>En)
        rhs .= -fem.A*yn .- fem.LM*max.(yn,0) .+ fem.M*u
        yn .+= (fem.A.+fem.LM*DN)\rhs
        Enew .= yn.>=0
        converged = Enew==En
        En .= Enew
    end
    return yn
end

"apply adjoint Bouligand derivative by solving 'adjoint' equation"
function Correction(fem::FEM,y,α,res)
    KyM = spdiagm(0=>(y.>=0))*fem.LM
    Cₙ = [fem.A.+KyM -fem.M; 1.0./α.*fem.M fem.AT.+KyM]
    dₙ = [zero(res);1.0./α.*fem.M*res]
    ξ  = Cₙ\dₙ
    return ξ[fem.ndof+1:end]
end

"apply modified Levenberg-Marquardt method to ydelta"
function modifiedLM(BLMparams,δ,yᵟ,uexact)
    u0,τ,α₀,r,maxit,fem = BLMparams
    cputime = @elapsed begin
    un = copy(u0)
    yn = F(fem,un)
    res = yᵟ .- yn
    resnorm = fem.norm(res)
    α = α₀
    BLMit  = 0
    @printf("It\tα\t\tresidual\trelative error\n")
    while (resnorm > τ*δ) & (BLMit <= maxit)
        BLMit +=1
        un .+= Correction(fem,yn,α,res)
        yn   = F(fem,un,yn)         # paper: F(fem,un) (no warmstarts)
        res .= yᵟ .- yn
        resnorm = fem.norm(res)
        errnorm = fem.relerror(un,uexact)
        α *= r  
        @printf("%d\t%1.2e\t%1.2e\t%1.2e\n",BLMit,α,resnorm,errnorm)
    end
    end
    if BLMit > maxit
        @printf("Failed to converge\n")
    else
        rate = fem.norm(un-uexact)/sqrt(δ)
        @printf("Estimated convergence rate %1.2f\n",rate)
        lograte = BLMit/(1+abs(log(δ)))
        @printf("Estimated logarithmic rate %1.2f\n",lograte)
        @printf("Elapsed CPU time: %f seconds\n",cputime)
    end
    return un
end

"compute exact solution and data with F nondifferentiable on 2β-measure set"
function exact_sol(fem,β)
    x,y = fem.x,fem.y
    chi = @. (x>=β)&(x<=1-β)
    yex = @. (x-β)^2*(x-1+β)^2*sin(2π*y)*chi
    uex = @. chi*(4π^2*yex-2*((2*x-1)^2+2*(x-β)*(x-1+β))*sin(2π*y))+max(yex,0)
    return yex,uex
end

"""
    run_example(N,δ,β)

Driver function for Bouligand-Levenberg-Marquardt iteration: Create data 
and compute reconstruction for an N×N discretization of the unit square, 
noise level δ, and exact parameter at which the forward mapping is 
nondifferentiable for a set of measure 2β
"""
function run_example(N,δ,β)
    # setup finite element discretization, construct exact and noisy data
    fem = FEM(N)
    yexact,uexact = exact_sol(fem,β)
    yᵟ = yexact .+ 1.5*δ*randn(fem.ndof)
    δ₂ = fem.norm(yexact-yᵟ)      # noise level in L2
    # initialize structure for Bouligand-Levenberg-Marquardt iteration
    α₀ = 1.0        # initial Tikhonov parameter for linearization
    r  = 0.5        # reduction factor for Tikhonov parameter
    τ  = 1.1        # parameter for discrepancy principle
    maxit = 100     # maximum number of iterations
    # starting point u0 = \bar u satisfying source condition
    u0 = @. uexact - (10*sin(π*fem.x)*sin(2π*fem.y)) 
    # apply modified Levenberg-Marquardt method
    @printf("noise level δ = %1.5e\n",δ₂)
    BLMparams = u0,τ,α₀,r,maxit,fem
    uN = modifiedLM(BLMparams,δ₂,yᵟ,uexact)
    return uN
end

end
