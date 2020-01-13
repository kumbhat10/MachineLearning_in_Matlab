function W = randInitializeWeightswithEpsilon(L_in, L_out, epsilon_init)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

%  One effective strategy for choosing epsilon_init is to base it on the number of units in the network. 
% A good choice of epsilon_init = sqrt( 6 / (Lin + Lout)),
% where Lin = Sl and Lout = Sl+1 are the number of units in the layers adjacent to current layer.i.e 400 & 10
epsilon_init = 0.12;  % sqrt(6 / (400+10)) = 0.121

W = rand(L_out, 1 + L_in) + 2*epsilon_init - epsilon_init;  %% 0 < x < 1 ==>    -c < x*2c - c < c

end
