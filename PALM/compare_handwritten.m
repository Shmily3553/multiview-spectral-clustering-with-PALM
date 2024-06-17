close all; clear all; 

%% load data
% hand written
% load data\multiplefeatures\mfeat-fac;
% load data\multiplefeatures\mfeat-fou;
% load data\multiplefeatures\mfeat-kar;
% load data\multiplefeatures\mfeat-mor;
% load data\multiplefeatures\mfeat-pix;
% load data\multiplefeatures\mfeat-zer;
load data\uci-digit.mat mfeat_fac mfeat_fou mfeat_kar truth;
view = 3;
cluster = 10;
G{1,1} = mfeat_fac;
G{1,2} = mfeat_fou;
G{1,3} = mfeat_kar;
Y = truth;

G = samesize(G, view);
G0 = G;

%% initialization
dim = 5;
itr = 20;
losses1 = zeros(itr, 1);
losses2 = zeros(itr, 1);
lambda = [0.1, 0.5, 1, 5, 10, 50, 100, 200];
% theta = 1;
% lambda = 1;
theta = [0.1, 0.5, 1, 5, 10, 50, 100, 200];
sigma = 1;

for i = 1:view
    n = size(G{i}, 1);
    D{i} = zeros(n, n);              % n*n
    % Adjacency matrix W
    XX = dot(G{i}',G{i}');
    W{i} = exp(-(bsxfun(@plus,XX,XX')-2*G{i}*G{i}')/(2*sigma^2));
    % diagonal matrix D
    for p = 1:n
        D{i}(p,p) = sum(W{i}(p,:));
    end
    % Laplace matrix L
    L{i} = D{i} - W{i};              % n*n
end
% consensus matrix
G_star = rand(size(G{i}));           % n*d
G_star_0 = G_star;

I = eye(n);                          % n*n

% pair matrix
A = zeros(dim, n);
% hand written
A(1,1) = 1;
A(1,3) = -1;
A(2,6) = 1;
A(2,12) = -1;
A(3,7) = 1;
A(3,13) = -1;
A(4,15) = 1;
A(4,16) = -1;
A(5,17) = 1;
A(5,18) = -1;

cms_cf = [];
cms_gd = [];

%% lambda parameter
for l = 1:size(lambda,2)
    for t = 1:size(theta,2)
        fprintf('----------lambda is %.1f, theta is %.1f--------\n', lambda(l), theta(t));

        G = G0;
        G_star = G_star_0;
        % closed-form solution
        disp('closed-form solution')
        for j = 1:itr
            [G, G_star] = closedform(view, lambda(l), theta(t), G, G_star, L, I, A);
        end
        G_cf = G_star;
        cm_cf = measure(G_cf, cluster, truth);
        cms_cf = [cms_cf; cm_cf];
        
        G = G0;
        G_star = G_star_0;
        % gradient descent
        disp('gradient descent solution')
        for j = 1:itr 
            [G, G_star] = gradientdescent(view, lambda(l), theta(t), G, G_star, L, A);
        end
        G_gd = G_star;
        cm_gd = measure(G_gd, cluster, truth);
        cms_gd = [cms_gd; cm_gd];
    end    
end

best_cf = max(cms_cf);
best_gd = max(cms_gd);
best = [best_cf; best_gd];

save('result\best_handwritten.mat', "best");

%% functions
function [G, G_star] = closedform(view, lambda, theta, G, G_star, L, I, A)
    % update G{k}
    for k = 1:view        
        mu = norm(L{k})+ lambda;
        grad_G = L{k}*G{k} + lambda.*(G{k}-G_star);
        P = G{k} - 1 / mu * grad_G;
        [U, ~, V] = svd(P,"econ");
        G{k} = U * V';        
    end
    % update G_star
    G_inv = inv(lambda*view.*I + theta.*A'*A);
    G_sum = zeros(size(G_star));
    for v = 1:view
        G_sum = G_sum + G{v};
    end
    G_star = lambda .* G_inv * G_sum; 
end

function [G, G_star] = gradientdescent(view, lambda, theta, G, G_star, L, A)
    % update G{k}
    for k = 1:view        
        mu = norm(L{k}) + lambda;
        grad_G = L{k}*G{k} + lambda.*(G{k}-G_star);
        P = G{k} - 1 / mu * grad_G;
        [U, ~, V] = svd(P,"econ");
        G{k} = U * V';        
    end
    % update G_star
    eta = view * lambda + theta * norm(A)^2;
    G_tmp = zeros(size(G_star));
    for v = 1:view
        G_tmp = G_tmp + G_star - G{v};
    end
    grad_G_star = lambda.*G_tmp + theta.*A'*A*G_star;
    G_star = G_star - 1 / eta * grad_G_star;  
end

% pick the least number of features and shape the data
function G = samesize(G, k)
    sz = zeros(k,1);
    for f = 1:k
        s = size(G{1,f},2);
        sz(f) = s;
    end
    sz_min = min(sz);
    for d = 1:k
        % delete the rest col
        m = size(G{1,d},2);
        if m > sz_min
            G{1,d}(:, sz_min+1:m) = [];
        end
    end
end

function eval = measure(G_star, cluster, truth)
    % eval = [Acc, MIhat, Purity, NMI, MI, RI, ARI, F1]
    results = [];
    for ii = 1:10
        pred = kmeans(G_star, cluster, 'MaxIter', 200);
        % [ACC, MIhat, Purity] = result;
        res = ClusteringMeasure(pred, truth);
        [F1, RI, ARI] = randindex(truth, pred);
        [NMI, MI] = AMI(truth, pred);
        result = [res, NMI, MI, RI, ARI, F1];
        results = [results; result];
    end
    eval = mean(results);
end