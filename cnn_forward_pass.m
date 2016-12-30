function [ L, inter ] = cnn_forward_pass( I, model, y )
% CNN Forward pass
% Conv -- relu -- maxpool -- FC -- output
% I : 32x32x3
% W1 : 3x3 x 3  x 10
% b1 : 1x10
% W2 : 3x3 x 10 x 4
% b2 : 1x4

W1 = model.W1; 
b1 = model.b1; 
W2 = model.W2; 
b2 = model.b2;   
W3 = model.W3;
b3 = model.b3; 
W4 = model.W4; 
b4 = model.b4;
stride = 2; 

% Convolution Layer-1
u1 = layer_conv( I, W1, b1 );

% ReLU
u2 = max( 0, u1 );

% Max Pool
[u3 u3_idx] = layer_maxpool( u2, stride );




% Conv - 2
u4 = layer_conv( u3, W2, b2 );

% ReLU
u5 = max( 0, u4 );

% Max Pool
[u6 u6_idx] = layer_maxpool( u5, stride );


% Fully Connected Layers
u6_r = reshape( u6, 1, prod( size(u6 ) ) );

u7 = u6_r * W3; 
u8 = u7 + b3;

u9 = max( 0, u8 );


u10 = u9 * W4;
u11 = u10 + b4;
u11 = u11 - max(u11);

% Softmax
L = SoftMaxLoss( u11, y );



inter.u1 = u1;
inter.u2 = u2;
inter.u3 = u3; inter.u3_idx = u3_idx;
inter.u4 = u4;
inter.u5 = u5;
inter.u6 = u6; inter.u6_r = u6_r; inter.u6_idx = u6_idx;
inter.u7 = u7;
inter.u8 = u8;
inter.u9 = u9;
inter.u10 = u10;
inter.u11 = u11;
inter.L = L;


end

