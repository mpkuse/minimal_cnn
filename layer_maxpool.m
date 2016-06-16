function [ outp, outpIndx ] = layer_maxpool( inp, stride )
% Max Pooling Layer
% inp : axbxf
% outp : a/2 x b/2 x f


outp = zeros( size(inp,1)/stride, size(inp,2)/stride, size(inp,3) );
outpIndx = zeros( size(inp,1)/stride, size(inp,2)/stride, size(inp,3) );
for kd = 1:size(inp,3)
    [outp(:,:,kd) outpIndx(:,:,kd)] = maxPoolIm( inp(:,:,kd), stride );
end

end

