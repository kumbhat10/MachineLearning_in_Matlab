function plotThetaCollected(X_collect,index)
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units

nn_params_collected = X_collect(:,index);
Theta1 = reshape(nn_params_collected(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
displayData(Theta1(:, 2:end));
title([  'iterations = ',num2str(index)])

end