close all; clear all; 

%% load data
% Gdata
load data\Gdata.mat;
view = 2;
cluster = 2;
truth = labels;

%% initialization
dim = 5;
itr = 20;
losses1 = zeros(itr, 1);
losses2 = zeros(itr, 1);
lambda = [0.1, 0.5, 1, 5, 10, 50, 100, 200];
theta = [0.1, 0.5, 1, 5, 10, 50, 100, 200];
% theta = 1;
% lambda = 1;
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
% Gdata
A(1,1) = 1;
A(1,7) = -1;
A(2,13) = 1;
A(2,16) = -1;
A(3,17) = 1;
A(3,18) = -1;
A(4,26) = 1;
A(4,29) = -1;
A(5,33) = 1;
A(5,35) = -1;

views_cf = [];
views_gd = [];

%% lambda parameter
% run for each view and then get the avg results
for v = 1:view
    G0 = G{1,v};
    L_ = L{1,v};
    cms_cf = [];
    cms_gd = [];

    for l = 1:size(lambda,2)
        for t = 1:size(theta,2)
            fprintf('----------lambda is %.1f, theta is %.1f--------\n', lambda(l), theta(t));
    
            G_ = G0;
            G_star = G_star_0;
            % closed-form solution - single view
            disp('closed-form solution')
            for j = 1:itr
                [G_, G_star] = closedform(lambda(l), theta(t), G_, G_star, L_, I, A);
            end
            G_cf = G_star;
            cm_cf = measure(G_cf, cluster, truth);
            cms_cf = [cms_cf; cm_cf];
            
            G_ = G0;
            G_star = G_star_0;
            % gradient descent - single view
            disp('gradient descent solution')
            for j = 1:itr 
                [G_, G_star] = gradientdescent(lambda(l), theta(t), G_, G_star, L_, A);
            end
            G_gd = G_star;
            cm_gd = measure(G_gd, cluster, truth);
            cms_gd = [cms_gd; cm_gd];
        end    
    end
    view_cf = max(cms_cf);
    view_gd = max(cms_gd);
    views_cf = [views_cf; view_cf];
    views_gd = [views_gd; view_gd];
end

avg = [mean(views_cf); mean(views_gd)];

save('result\SV\best_Gdata.mat', "avg");

%% functions
function [G, G_star] = closedform(lambda, theta, G, G_star, L, I, A)
    % update G
    mu = norm(L)+ lambda;
    grad_G = L*G + lambda.*(G-G_star);
    P = G - 1 / mu * grad_G;
    [U, ~, V] = svd(P,"econ");
    G = U * V';
    % update G_star
    G_inv = inv(lambda.*I + theta.*A'*A);
    G_star = lambda .* G_inv * G; 
end

function [G, G_star] = gradientdescent(lambda, theta, G, G_star, L, A)
    % update G
    mu = norm(L) + lambda;
    grad_G = L*G + lambda.*(G-G_star);
    P = G - 1 / mu * grad_G;
    [U, ~, V] = svd(P,"econ");
    G = U * V'; 
    % update G_star
    eta = lambda + theta * norm(A)^2;
    grad_G_star = lambda.*(G_star - G) + theta.*A'*A*G_star;
    G_star = G_star - 1 / eta * grad_G_star;  
end

function eval = measure(G_star, cluster, truth)
    % eval = [Acc, MIhat, Purity, NMI, MI, RI, ARI, F1]
    results = [];
    for ii = 1:10
        pred = kmeans(G_star, cluster);
        % [ACC, MIhat, Purity] = result;
        res = ClusteringMeasure(pred, truth);
        [F1, RI, ARI] = randindex(truth, pred);
        [NMI, MI] = AMI(truth, pred);
        result = [res, NMI, MI, RI, ARI, F1];
        results = [results; result];
    end
    eval = mean(results);
end