function [implbls,impl] = intervalDominance(mPred)
[lowexpectutil, uppexpectutil] = lowuppexputil(mPred);
dominate = zeros(length(lowexpectutil));
for i = 1 : length(lowexpectutil)
    for j = 1 : length(uppexpectutil)
        dominate(i,j) = lowexpectutil(i)> uppexpectutil(j);
    end
end
implbls = find(sum(dominate)==0);
k = log2(length(mPred)+1);
aa = zeros(k,1);
aa(implbls)=ones(length(implbls),1);
impl = bi2de(aa');
end

