function [mPred,BayesPred] = predEvMlp(nn_params,hidden_layer_size,covpost,xtst)
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
    
    

discreteResa = 20000;
a = linspace(-100,100,discreteResa);
a = sort([a,a3t]);


sigma_a = bt'*covpost./10*bt;
mu_a = a3t;

Na = normpdf(a,mu_a,sigma_a);
NormNa = Na./max(Na);


ha = sigmoid(a);
BayesPred = sum(Na./sum(Na).*ha);

[hauniq, indexuniq] = unique(ha); 
queryha = sort([0.001:0.001:0.999,z3t]);
plxt = interp1(hauniq, NormNa(indexuniq),queryha);
plxt = [0,plxt,0];
pixt = [0,queryha,1];


% figure;
% %plot(pixt,plxt,'-k','LineWidth',1);
% plot(pixt,plxt,'-k','LineWidth',1);
% xlabel('omega');ylabel('pl(omega)');
% axis([0 1 0 1]);

pixthat=z3t;


pixtinf = pixt(pixt<=pixthat & pixt >=0);
plinf = plxt(pixt<=pixthat & pixt >=0);
pixtsup = pixt(pixt>=pixthat & pixt <=1);
plsup = plxt(pixt>=pixthat & pixt <= 1);

areainf = trapz(pixtinf,plinf);
areasup = trapz(pixtsup,plsup);

m1 = pixthat - areainf;
m0 = (1 - pixthat) - areasup;
m01 = areainf+areasup;
mPred = [m0 m1 m01];


end

