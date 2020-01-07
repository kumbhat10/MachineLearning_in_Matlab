function predict_accuracy = predict_Accuracy(X_collect,y, X)
predict_accuracy = zeros(size(X_collect,2),1);
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;

for i = 1:size(X_collect,2)
    nn_params_collected = X_collect(:,i);
    Theta1 = reshape(nn_params_collected(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params_collected((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_labels, (hidden_layer_size + 1));
    
    pred = predict(Theta1, Theta2, X);
    predict_accuracy(i,1) = mean(double(pred == y)) * 100 ;
    % fprintf('\nTraining Set Accuracy: %0.3f\n', mean(double(pred == y)) * 100);
end
end