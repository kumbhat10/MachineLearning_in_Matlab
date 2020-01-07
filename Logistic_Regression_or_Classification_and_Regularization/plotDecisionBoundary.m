function plotDecisionBoundary(theta, X, y,degree)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the
%   positive examples and o for the negative examples. X is assumed to be
%   a either
%   1) Mx3 matrix, where the first column is an all-ones column for the
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    
    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
    
    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    %     axis([30, 100, 30, 100])
    xlim([min(X(:,2))*1.1, max(X(:,2))*1.1])
    ylim([min(X(:,3))*1.1, max(X(:,3))*1.1])
    
else
    % Here is the grid range
    %     u = linspace(30, 100, 100);
    %     v = linspace(30, 100, 100);
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
end
hold off

end
