function [H] = OuterProdHessian(nn_params, input_layer_size, hidden_layer_size,X)
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),1, (hidden_layer_size + 1));
n = size(X, 1);

% Feedforward propagation
% input layer
z1 = [ones(n,1) X];
a1 = z1;
% hidden layer
a2 = z1*W1';
z2 = sigmoid(a2);
z2 = [ones(n,1) z2];
% output layer
a3 = z2*W2';
z3 = sigmoid(a3);

H = zeros(length(nn_params));
for t = 1:n
    z1t = z1(t,:);
	z2t = z2(t,:);
    z3t = z3(t,:);
    
    dha2 = z2t(2:end) .* (1-z2t(2:end));
    A = W2(2:end) .* dha2;
    B = repmat(A,input_layer_size+1,1);
    C = repmat(z1t',1,hidden_layer_size);
    D = B.*C;
    
    btW1 = D;
    btW2 = z2t;
    
    bt = [btW1(:);btW2(:)];
    
    Ht = (z3t*(1-z3t)).*(bt*bt');
    H = H + Ht;
end

% check validity of outer product approximation of hessian
tf = issymmetric(H);
d = eig(H);
isposdef = all(d > 0);


end


