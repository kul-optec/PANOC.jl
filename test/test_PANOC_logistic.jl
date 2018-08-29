using Test
using PANOC
using ProximalOperators

using Random

Random.seed!(0)

@testset "L1-LogReg" begin
    T = Float64
    A = T[
        1.0  -2.0   3.0  -4.0  5.0;
        2.0  -1.0   0.0  -1.0  3.0;
        -1.0   0.0   4.0  -3.0  2.0;
        -1.0  -1.0  -1.0   1.0  3.0
    ]
    b = T[1.0, 2.0, 3.0, 4.0]
    m, n = size(A)
    f = Translate(LogisticLoss(ones(m)), -b)
    lam = 0.1
    g = NormL1(lam)
    x_star = [0, 0, 2.114635341704963e-01, 0, 2.845881348733116e+00]

    @testset "Forward-backward envelope" begin
        Lf = opnorm(A)^2
        gamma = 0.95/Lf
        for x in [[zeros(T, 5)]; [randn(T, 5) for k=1:4]]
            grad_f_Ax, f_Ax = gradient(f, A*x)
            F_x = f_Ax + g(x)
            At_grad_f_Ax = A'grad_f_Ax
            z, g_z = prox(g, x - gamma * At_grad_f_Ax, gamma)
            res = x - z
            FBE_x = PANOC.f_model(f_Ax, At_grad_f_Ax, res, gamma) + g_z
            F_z = f(A*z) + g_z
            tol = 10*eps(T)*(1+abs(FBE_x))
            @test T(0) <= F_x < T(Inf)
            @test T(0) <= F_z < T(Inf)
            @test T(0) <= FBE_x < T(Inf)
            @test FBE_x <= F_x - (0.5/gamma)*norm(res)^2 + tol
            @test F_z <= FBE_x - (0.5/gamma)*(1 - gamma*Lf)*norm(res)^2 + tol
        end
    end

    @testset "PANOC(adaptive)" begin
        for x0 in [[zeros(T, 5)]; [randn(T, 5) for k=1:4]]
            x_panoc, it_panoc = panoc(f, A, g, x0, maxit=1000, tol=1e-8, verbose=false)
            @test x_panoc ≈ x_star
            @test it_panoc <= 50
        end
    end
end
