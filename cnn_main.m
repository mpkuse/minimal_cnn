clear all
close all

%% Data
[ fileNames, classInx ] = textread( './data/test_batch.bin_dir/annotation.txt', '%s %d' ); 

%% Init Model
% rng( 'shuffle' );
% model.W1 = randn( 3,3,  3, 10 ); model.b1 = randn( 1, 10 );
% model.W2 = randn( 3,3, 10, 4  ); model.b2 = randn( 1, 4 );
% model.W3 = randn( 256,32 );      model.b3 = randn( 1, 32 );
% model.W4 = randn( 32, 10 );      model.b4 = randn( 1, 10 );
load model

imName = sprintf( '%s/%s', './data/', fileNames{21} );
display( imName );
im = im2double( imread( imName ) );
y = zeros( 1, 10 ); y( classInx(21)+1 ) = 1;

%% Iterations
step = 0.5;
for itr = 1:10
    %tic
    [L, inter] = cnn_forward_pass( im, model, y );
    display( sprintf( '\n%d:%f', itr, L ) );
    
    
    grad = cnn_back_prop( im, model, inter, y );
    
    %step = [ max( grad.W1(:) ), max( grad.W1(:) ), max( grad.W1(:) ), ;
    step = [max( grad.W1(:) ), max( grad.b1(:) ),max( grad.W2(:) ),max( grad.b2(:) ),  max( grad.W3(:) ),max( grad.b3(:) ),max( grad.W4(:) ),max( grad.b4(:) )];
    step = .1 ./ step;
      display( sprintf( '(%.e) %.2e, %.2e,  %.2e, %.2e,  %.2e, %.2e,  %.2e, %.2e', step(1),max( model.W1(:) ), max( model.b1(:) ),max( model.W2(:) ),max( model.b2(:) ),  max( model.W3(:) ),max( model.b3(:) ),max( model.W4(:) ),max( model.b4(:) ) ) );
    
    %modelOut = PLUS( model, grad, step );
    %model = modelOut;
    
    lambda = 0.0;
    model.W1 = model.W1 - step(1)*( grad.W1 + lambda*model.W1 );
    model.b1 = model.b1 - step(2)*( grad.b1 + lambda*model.b1 );
    
    model.W2 = model.W2 - step(3)*( grad.W2 + lambda*model.W2 );
    model.b2 = model.b2 - step(4)*( grad.b2 + lambda*model.b2 );
    
    model.W3 = model.W3 - step(5)*( grad.W3 + lambda*model.W3 );
    model.b3 = model.b3 - step(6)*( grad.b3 + lambda*model.b3 );
    
    model.W4 = model.W4 - step(7)*( grad.W4 + lambda*model.W4 );
    model.b4 = model.b4 - step(8)*( grad.b4 + lambda*model.b4 );
    
    %toc
end