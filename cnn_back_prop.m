function [ output_args ] = cnn_back_prop(  im, model, Q, y )

% note: dL is implicit. To clarify, whenevr du11 is written, 
% it actually means dL_du11.

epsilon = 1E-8;
exp_u = exp( Q.u11 - max(Q.u11) );
sigma = sum( exp_u + epsilon );
p = -log( exp_u + epsilon / sigma );
du11 = p - y; %1x10 



db4 = du11; %1x10 NEED


du10 = du11; %1x10


dw4 = Q.u9' * du10; %32x10 NEED
du9 = du10 * model.W4'; %1x32

du8 = du9 * diag(max( 0, Q.u8 ) > 0); %1x32


db3 = du8; %1x32 NEED
du7 = du8; %1x32

dw3 = Q.u6_r' * du7; %256x32 NEED
du6_r = du7 * model.W3'; %1x256, can be reshapped if need be
du6 = reshape( du6_r, 8,8,4 ); %8x8x4 

% derivative of maxpool layer (thinking similar to the ReLU), max gets it
% all, this is where the max index stored in maxpool layer comes to play
du5 = backprop_maxpool( du6, Q.u6, Q.u6_idx, 2 );

% derivative of reLU
du4 = max( 0, du5);


% derivative of convolution layer. Equations taken from deeplearning book.
% But beware that the index-notations are different. In particular we use
% the 1st 2 indx as spatial index, whereas the book uses last 2 index for
% spatial. 
[du3 dW2] = backprop_conv( Q.u3, model.W2, Q.u4, du4 );

end

