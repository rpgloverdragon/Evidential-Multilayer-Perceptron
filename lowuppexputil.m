function [lowexpectutil, uppexpectutil] = lowuppexputil(mPred)
k=log2(size(mPred,2)+1);
util = eye(k);
lowexpectutil = zeros(k,1);
uppexpectutil = zeros(k,1);
for fi = 1 : k
    expectmin=0;expectmax=0;
    for i = 1 : 2^k-1
        idi = find(de2bi(i,k));
        utilmin = min(util(idi,fi));
        utilmax = max(util(idi,fi));
        expectmin=expectmin+mPred(i)*utilmin;
        expectmax=expectmax+mPred(i)*utilmax;
    end
    lowexpectutil(fi)=expectmin;
    uppexpectutil(fi)=expectmax;
end
