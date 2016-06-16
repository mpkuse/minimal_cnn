function [ outIm outImIndx] = maxPoolIm( im, stride )
% Max pool an image (1-channel)

outIm = zeros( size(im) / stride );
outImIndx = zeros( size(im) / stride );
for i=1:stride:size(im,1)
    for j=1:stride:size(im,2)
        %display( sprintf( '%d,%d --> %d,%d',  i,i+stride-1, j,j+stride-1 ) );
        subIm = im( i:i+stride-1, j:j+stride-1 );
        [C Idx] = max( subIm(:) ); %index will be in col-major
        
        outIm( ceil(i/2), ceil(j/2) ) = C; 
        
        % need also to store which index gave the maximum
        outImIndx( ceil(i/2), ceil(j/2) ) = Idx; %later ind2sub() may be useful function
    end
end

end

