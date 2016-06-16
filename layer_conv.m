function [ u ] = layer_conv( I, W, b )
% Convolution Layer
% I : axbxc
% W : mxn xcxf
% b : 1xf
% u : axbxf

u = zeros( size(I,1), size(I,2), size(W,4) );
for kd = 1:size(W,4)
    for k = 1:size(W,3)
        u(:,:,kd) = u(:,:,kd) + conv2(  I(:,:,k), W(:,:,k, kd), 'same' );
    end
    u( :,:, kd ) = u( :,:, kd ) + b(kd);
end

end

