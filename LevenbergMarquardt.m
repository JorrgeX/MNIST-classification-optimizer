function [w,fvals,gnorm] = LevenbergMarquardt(r_and_J,w,kmax,tol)

fsz = 20; % fontsize
iter_max = kmax; % the maximal number of iterations

%% parameters for trust region
Delta_max = 1; % the max trust-region radius
Delta_min = 1e-14; % the minimal trust-region radius
Delta = 0.2; % the initial radius
eta = 0.01; % step rejection parameter
subproblem_iter_max = 5; % the max # of iteration for quadratic subproblems
tol_sub = 1e-1; % relative tolerance for the subproblem
rho_good = 0.75;
rho_bad = 0.25;

%% Set up the initial configuration
[rk, Jk] = r_and_J(w);
gnorm = zeros(kmax, 1);
fvals = zeros(kmax, 1);

%% start minimization
g = Jk'*rk;
norm_g = norm(g);
norm_r = norm(rk);
f = 0.5*norm_r.^2;
fprintf("Initially, f = %d, || g || = %d\n",f, norm_g);
iter = 1;

[n_row, n_col] = size(Jk);
I = eye(n_col);

while norm_g > tol && iter < iter_max
    fvals(iter) = f;
    gnorm(iter) = norm_g;

    % do Tikhonov regularization for the case J is rank-deficient
    B = Jk'*Jk + (1e-6)*I;
    flag_boundary = 0;
    % check if B is SPD
    eval_min = min(eig(B));
    j_sub = 0;
    if eval_min > 0 % B is SPD: B = R'*R, R'*R*p = -g 
        p = -B\g;
        p_norm = norm(p);
        if p_norm > Delta % else: we are done with solbing the subproblem
            flag_boundary = 1;
        end
    else
        flag_boundary = 1;
    end

    % pstar = -B\g; % unconstrained minimizer
    % if norm(pstar) <= Delta
    %     p = pstar;

    if flag_boundary == 1 % solve constrained minimization problem
        % lam = 1; % initial guess for lambda
        lambda_min = max(-eval_min,0);
        lam = lambda_min + 1;
        flag_subproblem_success = 0;
        while j_sub < subproblem_iter_max
            j_sub = j_sub + 1;
            B1 = B + lam*I;
            C = chol(B1); % do Cholesky factorization of B
            p = -C\(C'\g); % solve B1*p = -g
            np = norm(p);
            dd = abs(np - Delta); % R is the trust region radius
            if dd < tol_sub*Delta
                flag_subproblem_success = 1;
                break
            end
            q = C'\p; % solve C^\top q = p
            nq = norm(q);
            lamnew = lam + (np/nq)^2*(np - Delta)/Delta;
            if lamnew > lambda_min
                lam = lamnew;
            else
                lam = 0.5*(lam + lambda_min);
            end
        end
        if flag_subproblem_success == 0
            p = cauchy_point(B,g,Delta);
        end
    end

    % assess the progress
    wnew = w + p;
    [rk1, Jk1] = r_and_J(wnew);
    fnew = 0.5*norm(rk1).^2;
    gnew = Jk1'*rk1;
    
    regularized_JJ = Jk'*Jk + (1e-6)*I;
    mnew = f + p'*Jk'*rk + 0.5*p'*regularized_JJ*p;
    rho = (f - fnew+1e-14)/(f - mnew+1e-14);
    % adjust the trust region
    if rho < rho_bad
        Delta = max([0.25*Delta,Delta_min]);
    else
        if rho > rho_good 
            Delta = min([Delta_max,2*Delta]);
        end
    end
    % accept or reject step
    if rho > eta            
        w = wnew;
        f = fnew;
        g = gnew;
        norm_g = norm(g);
        rk = rk1;
        Jk = Jk1;
        fprintf('Accept: iter # %d: f = %.10f, |df| = %.4e, rho = %.4e, Delta = %.4e, j_sub = %d\n',iter,f,norm_g,rho,Delta,j_sub);
    else
        fprintf('Reject: iter # %d: f = %.10f, |df| = %.4e, rho = %.4e, Delta = %.4e, j_sub = %d\n',iter,f,norm_g,rho,Delta,j_sub);
    end

    iter = iter + 1;
end
end

function p = cauchy_point(B,g,R)
    ng = norm(g);
    ps = -g*R/ng;
    aux = g'*B*g;
    if aux <= 0
        p = ps;
    else
        a = min(ng^3/(R*aux),1);
        p = ps*a;
    end
end