function [w,fvals,gnorm] = StochasticNesterov(q_loss,grad_qloss,w,len,kmax,tol)

fsz = 20; % fontsize

%% parameters for SGD
stepsize = 0.9;
batchsize = 67;
lam = 0.001;
I = 1:len;

%% Set up the initial configuration
randomIndices = randperm(len, batchsize);

gnorm = zeros(kmax, 1);
fvals = zeros(kmax, 1);

%% start minimization
f = q_loss(I,w,lam);
g = grad_qloss(I,w,lam);
norm_g = norm(g);
g_subset = grad_qloss(randomIndices,w,lam)/length(randomIndices);
fprintf("Initially, f = %d, || g || = %d\n",f, norm_g);
iter = 1;

% iter = 1, use SGD
fvals(iter) = f;
gnorm(iter) = norm_g;
wold = w; % store old value
w = w - stepsize*g_subset;
f = q_loss(I,w,lam);
g = grad_qloss(I,w,lam);
norm_g = norm(g);
iter = iter + 1;

% start StochasticNesterov
while norm_g > tol && iter < kmax
    fvals(iter) = f;
    gnorm(iter) = norm_g;

    mu = 1 - (3/(5+iter));
    yk = (1+mu)*w - mu*wold;
    randomIndices = randperm(len, batchsize);
    grad_yk = grad_qloss(randomIndices,yk,lam)/length(randomIndices);
    wold = w;
    w = yk - stepsize*grad_yk;
    f = q_loss(I,w,lam);
    g = grad_qloss(I,w,lam);
    norm_g = norm(g);

    if mod(iter ,10)==0
        fprintf('iter = %d, a = %d, f = %d, ||g|| = %d\n',iter,stepsize,fvals(iter),gnorm(iter));
    end

    iter = iter + 1;
end

end
