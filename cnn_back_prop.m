function [ grad ] = cnn_back_prop(  im, model, Q, y )
% Back propagation for my network. Returns the grad of loss function 
% wrt to every optimization variable.
% note: dL is implicit. To clarify, whenevr du11 is written, 
% it actually means dL_du11.

epsilon = 1E-8;
exp_u = exp( Q.u11  );
sigma = sum( exp_u + epsilon );
p = -log( exp_u + epsilon / sigma );
du11 = p - y; %1x10 



db4 = du11; %1x10 NEED


du10 = du11; %1x10


dW4 = Q.u9' * du10; %32x10 NEED
du9 = du10 * model.W4'; %1x32

du8 = du9 * diag(max( 0, Q.u8 ) > 0); %1x32


db3 = du8; %1x32 NEED
du7 = du8; %1x32

dW3 = Q.u6_r' * du7; %256x32 NEED
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
[dW2, du3, db2] = backprop_conv( Q.u3, model.W2, Q.u4, du4, [1 1 1] );


% grad of maxpool-1
du2 = backprop_maxpool( du3, Q.u3, Q.u3_idx, 2 );


%grad of reLU
du1 = max( 0, du2 );

%backprop-conv
[dW1 , ~, db1] = backprop_conv( im, model.W1, Q.u1, du1, [1 0 1] );


% in grad.__ we do not write d, (eg dW1) but we mean it
grad.W1 = dW1;
grad.b1 = db1;
grad.W2 = dW2;
grad.b2 = db2;
grad.b3 = db3;
grad.W3 = dW3;
grad.b4 = db4;
grad.W4 = dW4;

display( sprintf( '(grad-norm) %e, %e,   %e, %e,   %e, %e,   %e, %e', norm(dW1(:)), norm(db1(:)),    norm(dW2(:)), norm(db2(:)),    norm(dW3(:)), norm(db3(:)),    norm(dW4(:)), norm(db4(:))   ) );
display( sprintf( '(grad-max ) %.2e, %.2e,  %.2e, %.2e,  %.2e, %.2e,  %.2e, %.2e',max( grad.W1(:) ), max( grad.b1(:) ),max( grad.W2(:) ),max( grad.b2(:) ),  max( grad.W3(:) ),max( grad.b3(:) ),max( grad.W4(:) ),max( grad.b4(:) ) ) );
display( sprintf( '(grad-min) %.2e, %.2e,  %.2e, %.2e,  %.2e, %.2e,  %.2e, %.2e',min( grad.W1(:) ), min( grad.b1(:) ),min( grad.W2(:) ),min( grad.b2(:) ),  min( grad.W3(:) ),min( grad.b3(:) ),min( grad.W4(:) ),min( grad.b4(:) ) ) );
  
end

