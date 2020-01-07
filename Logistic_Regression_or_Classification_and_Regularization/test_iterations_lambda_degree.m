% data = load('ex2data1.txt');
data = load('ex2data2.txt');
lambda = 0;
degree = 3;
%%
X = data(:, [1, 2]); y = data(:, 3);
X = mapFeature(X(:,1), X(:,2),degree);
initial_theta = zeros(size(X, 2), 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

%%
plotDecisionBoundary(theta, X, y,degree);
hold on;
%%
%{
u = linspace(min(X(:,2))*1.1, max(X(:,2))*1.1, 100);
v = linspace(min(X(:,2))*1.1, max(X(:,2))*1.1, 100);
z = zeros(length(u), length(v));
for i = 1:length(u)
    for j = 1:length(v)
        z(i,j) = mapFeature(u(i), v(j),degree)*theta;
    end
end
z = z'; % important to transpose z before calling contour
% Notice you need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)
%}
