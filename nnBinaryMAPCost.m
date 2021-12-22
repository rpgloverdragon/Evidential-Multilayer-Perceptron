function [J, grad] = nnBinaryMAPCost(nn_params, input_layer_size, hidden_layer_size, X, y, lambda)
% Implements the neural network cost function for a two layer neural network which performs classification
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),1, (hidden_layer_size + 1));
n = size(X, 1);
        

% Feedforward propagation
a1 = [ones(n,1) X];
a2 = sigmoid(a1*Theta1');
a2 = [ones(n,1) a2];	
a3 = sigmoid(a2*Theta2');
J = -sum(y.*log(a3) + (1 - y).*log(1 - a3));

regTheta1 =  Theta1(:,2:end);
regTheta2 =  Theta2(:,2:end);

error = (lambda/2) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));

% Back propagation
J = J + error;
del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));
for t = 1:n
	a1t = a1(t,:);
	a2t = a2(t,:);
	a3t = a3(t,:);
	yt = y(t,:);
	d3 = a3t - yt;
	d2 = Theta2'*d3' .* sigmoidGradient([1;Theta1 * a1t']);
	del1 = del1 + d2(2:end)*a1t;
	del2 = del2 + d3' * a2t;
end
Theta1_grad =  del1 + (lambda)*[zeros(size(Theta1, 1), 1) regTheta1];
Theta2_grad =  del2 + (lambda)*[zeros(size(Theta2, 1), 1) regTheta2];


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
