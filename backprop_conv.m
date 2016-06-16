function [ dK dV db] = backprop_conv( V, K, Z, G, selectn )
% Back propagation for a convolution layer
% V : input to convolution layer (axbxc)
% K : kernel (mxn x cxf)
% Z : output of convolution layer (axbxf)
% G : gradient of loss function wrt Z (axbxf)
% selectn : 3-vector. [1 1 1] will compute all 3 derivatives, [1 0 0] will
%             compute just dK. [ 1 0 1] will compute just dK and db.
% outputs
% dK : gradient of loss function wrt kernels
% dV : gradient of loss function wrt input images

% Note:
% Equations taken from deeplearning book.
% But beware that the index-notations are different. In particular we use
% the 1st 2 indx as spatial index, whereas the book uses last 2 index for
% spatial.

%%
dK = zeros( size(K) );
dV = zeros( size(V) );
db = zeros( 1, size(Z,3) );


if selectn(1) == 1
pad_V = padarray( V, [1 1] );

% computation of dK
% See formula in deep-learning book (Chp 9, Pg 28-29)
for i=1:size(G,3)
    
    for j=1:size(V,3)
        
        for k=1:size(K,1)
            
            for l=1:size(K,1)
                
                
                
                for m=1:size(V,1)
                    for n=1:size(V,2)
                        dK( k,l, j, i ) = dK( k,l, j, i ) + G( m,n, i ) * pad_V( m-1+k, n-1+l );
                    end
                end
                
                
                
            end
        end
    end
    
end

end

%%
% computation of dV
% See formula in deep-learning book (Chp 9, Pg 28-29)
if selectn(2) == 1
for i=1:size(V,3) %# of input channels
    for j=1:size(V,1) %spatial 
        for k=1:size(V,2) %spatial
            
            
            
            for l=1:size(G,1)
                for n=1:size(G,2)
                    m = j - (l-1);
                    p = k - (n-1);
                    if( m > 0 && p>0 && m<size(K,1) && p<size(K,2) )
                    
                    for q=1:size(G,3)
                        dV( j,k, i ) = dV( j,k, i ) + K( m,p, i,q ) * G( l,n, q );
                    end
                    end

                    
                end
            end
        
            
            
            
             
            
        end
    end
end
end

%%
% computation of db (gradient of loss function with respect to bias)
% dL_db_{i} = \sum_{\forall j,k} dL_dZ_{ijk} . dZ_{ijk}_db_{i} %note last 2 indx r spatial indx
%           = \sum_{\forall j,k} dL_dZ_{ijk} 
if selectn(3) == 1
db = zeros( 1, size(Z,3) );
for i=1:length(db)
    db(i) = sum( sum( Z(:,:,i) ));
end


end

