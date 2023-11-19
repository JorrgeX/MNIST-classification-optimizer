function [w,fvals,gnorm] = StochasticAdam(q_loss,grad_qloss,w,len,kmax,tol)

fsz = 20; % fontsize

%% parameters for Stochastic Adam
stepsize = 0.9;
batchsize = 67;
b1 = 0.9;
b2 = 0.999;
epsilon = 1e-8;
eta = 0.001;
m = zeros(length(w), 1);
% v = zeros(length(w), 1);
v = 0;
lam = 0.001;
I = 1:len;

%% Set up the initial configuration
gnorm = zeros(kmax, 1);
fvals = zeros(kmax, 1);

%% start minimization
f = q_loss(I,w,lam);
g = grad_qloss(I,w,lam);
norm_g = norm(g);
fprintf("Initially, f = %d, || g || = %d\n",f, norm_g);
iter = 1;

% start StochasticNesterov
while norm_g > tol && iter < kmax
    fvals(iter) = f;
    gnorm(iter) = norm_g;
    
    randomIndices = randperm(len, batchsize);
    g_subset = grad_qloss(randomIndices,w,lam)/length(randomIndices);
    m = b1*m + (1-b1)*g_subset;
    % v = b2*v + (1-b2)*(g_subset.*g_subset);
    v = max(b2*v, norm(g_subset));
    % m_hat = m/(1-b1.^(iter+1));
    % v_hat = v/(1-b2.^(iter+1));

    % w = w - (eta/(sqrt(v_hat)+epsilon))*m_hat; 
    w = w - (stepsize/(1 - b1.^iter))*m/v;

    f = q_loss(I,w,lam);
    g = grad_qloss(I,w,lam);
    norm_g = norm(g);

    if mod(iter ,10)==0
        fprintf('iter = %d, f = %d, ||g|| = %d\n',iter,fvals(iter),gnorm(iter));
    end

    iter = iter + 1;
end

end