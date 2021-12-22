function [BayesPred] = predBayesMlp(nn_params,hidden_layer_size,covpost,xtst)
input_layer_size = size(xtst,2);
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));
W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),1, (hidden_layer_size + 1));

% Feedforward propagation
% input layer
z1t = [1 xtst];
a1t = z1t;
% hidden layer
a2t = z1t*W1';
z2t = sigmoid(a2t);
z2t = [1 z2t];
% output layer
a3t = z2t*W2';
z3t = sigmoid(a3t);

% BackProp
dha2 = z2t(2:end) .* (1-z2t(2:end));
A = W2(2:end) .* dha2;
B = repmat(A,input_layer_size+1,1);
C = repmat(z1t',1,hidden_layer_size);
D = B.*C;

btW1=D;
btW2=z2t;

bt = [btW1(:);btW2(:)];
    
    

discreteResa = 1000;
a = linspace(-50,50,discreteResa);
a = sort([a,a3t]);


sigma_a = bt'*covpost*bt;
mu_a = a3t;
Na = normpdf(a,mu_a,sigma_a);
ha = sigmoid(a);
BayesPred = [1-sum(Na./sum(Na).*ha), sum(Na./sum(Na).*ha)];


end

