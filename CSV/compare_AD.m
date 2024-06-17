close all; clear all; 

% %% generate data
% % obtain needed data from xls file and save into .mat files separately
% [~, id1] = xls2mat('data_for_MICCAI_JntRegCla.xlsx', 2);
% [~, id2] = xls2mat('CanSNPs_Top40Genes_org.xlsx', 1);
% data4 = reorder(id1, id2);

% preprocess data
view = 3;
cluster = 3;
% [G, ~, truth] = preprocess(view, data4);
% save data\AD.mat G truth;
load data\AD.mat;

G = concatviews(G, view);  % concatenate all views (n x view*m)
G0 = G;

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

n = size(G, 1);
D = zeros(n, n);              % n*n
% Adjacency matrix W
XX = dot(G',G');
W = exp(-(bsxfun(@plus,XX,XX')-2*G*G')/(2*sigma^2));
% diagonal matrix D
for p = 1:n
    D(p,p) = sum(W(p,:));
end
% Laplace matrix L
L = D - W;              % n*n
% consensus matrix
G_star = rand(size(G));           % n*d
G_star_0 = G_star;

I = eye(n);                          % n*n

% pair matrix
A = zeros(dim, n);
% AD
A(1,2) = 1;
A(1,5) = -1;
A(2,29) = 1;
A(2,30) = -1;
A(3,42) = 1;
A(3,57) = -1;
A(4,61) = 1;
A(4,71) = -1;
A(5,38) = 1;
A(5,35) = -1;

cms_cf = [];
cms_gd = [];

%% lambda parameter
for l = 1:size(lambda,2)
    for t = 1:size(theta,2)
        fprintf('----------lambda is %.1f, theta is %.1f--------\n', lambda(l), theta(t));

        G = G0;
        G_star = G_star_0;
        % closed-form solution - concat
        disp('closed-form solution')
        for j = 1:itr
            [G, G_star] = closedform(lambda(l), theta(t), G, G_star, L, I, A);
        end
        G_cf = G_star;
        cm_cf = measure(G_cf, cluster, truth);
        cms_cf = [cms_cf; cm_cf];
        
        G = G0;
        G_star = G_star_0;
        % gradient descent - concat
        disp('gradient descent solution')
        for j = 1:itr 
            [G, G_star] = gradientdescent(lambda(l), theta(t), G, G_star, L, A);
        end
        G_gd = G_star;
        cm_gd = measure(G_gd, cluster, truth);
        cms_gd = [cms_gd; cm_gd];
    end    
end

best_cf = max(cms_cf);
best_gd = max(cms_gd);
best = [best_cf; best_gd];

save('result\CSV\best_AD.mat', "best");

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

function G_concat = concatviews(G, view)
    G_concat = [];
    for v = 1:view
        G_concat = [G_concat, G{1,v}];
    end
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