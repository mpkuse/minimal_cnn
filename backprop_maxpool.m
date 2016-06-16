function [ d ] = backprop_maxpool( g, u, u_idx, stride )
% computes derivative for the maxpool layer. 
% g : gradient of loss function wrt to output of maxpool layer. 
% u : output of maxpool layer
% u_idx : output of maxpool layer, storing index of the max value

d = zeros( size(g,1)*stride, size(g,2)*stride, size(g,3) );

for k=1:size(g,3)
    d(:,:,k) = maxx( g(:,:,k ), u_idx, stride );
end

end


function [d] = maxx( a, a_idx, stride )

d = zeros( size(a)*2 );
s = stride;
for i=1:size(a,1)
    for j=1:size(a,2)
        [l m] = ind2sub( [stride stride], a_idx(i,j) );
        
        d( s*(i-1)+l, s*(j-1)+m ) = a( i, j );
    end
end

end