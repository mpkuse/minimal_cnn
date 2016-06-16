clear all
close all

%% Data
[ fileNames classInx ] = textread( './data/test_batch.bin_dir/annotation.txt', '%s %d' ); 

%% Init
model.W1 = randn( 3,3,  3, 10 ); model.b1 = randn( 1, 10 );
model.W2 = randn( 3,3, 10, 4  ); model.b2 = randn( 1, 4 );
model.W3 = randn( 256,32 );      model.b3 = randn( 1, 32 );
model.W4 = randn( 32, 10 );      model.b4 = randn( 1, 10 );

imName = sprintf( '%s/%s', './data/', fileNames{21} );
display( imName );
im = im2double( imread( imName ) );
y = zeros( 1, 10 ); y( classInx(21)+1 ) = 1;

tic
%cnn_forward_pass( im, W1, b1, W2, b2, W3, b3, W4, b4, y )
[L inter] = cnn_forward_pass( im, model, y )
cnn_back_prop( im, model, inter, y );

toc