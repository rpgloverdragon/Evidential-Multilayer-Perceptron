rng(1); N=30; % number of instances per class

% Generation of training instances with 2 classes
mu1 =[0, 0] ; mu2=[3,3];
S1=[2 0;0 2]; S2=[2 0;0 2]; 
r1 = mvnrnd(mu1,S1,N); r2 = mvnrnd(mu2,S2,N); 
Xfeat = [r1;r2];
ds= [ Xfeat ,[zeros(N,1);ones(N,1)]];
ds = ds(randperm(2*N),:);

% Definition of the Multi Layer Perceptron (MLP) structure
input_layer_size  = 2; 
hidden_layer_size = 3;

% Initialization of the MLP weights
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, 1);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)]; % Unroll parameters

% Learn the MLP by finding its MAP
alpha = 0.04; 
options = optimoptions('fmincon','Algorithm','interior-point','MaxIterations',1000000,'SpecifyObjectiveGradient',true,'CheckGradient',false,'Display','iter');
costFunction = @(p) nnBinaryMAPCost(p, input_layer_size, hidden_layer_size, ds(:,1:end-1), ds(:,end), alpha);
[nn_params_map, J_nn_params_map, ~,~,~,~,ADiff] = fmincon(costFunction, initial_nn_params,[],[],[],[],[],[],[], options);
Hmap = OuterProdHessian(nn_params_map, input_layer_size, hidden_layer_size, ds(:,1:end-1));
AOutprod = alpha.*eye(length(nn_params_map)) + Hmap;
covpost = inv(AOutprod);

% Predict the class of a test instance (predictive probability and mass function)
xt = [0,4];
[pit] = predBayesMlp(nn_params_map,hidden_layer_size,covpost,xt);
[mt, ~] =predEvMlp(nn_params_map,hidden_layer_size,covpost,xt); 

% Decisions based on bayesian and interval dominance strategies with 0/1 cost
[~, decibayes] = max(pit,[],2);
[deciid, ~] = intervalDominance(mt);  


