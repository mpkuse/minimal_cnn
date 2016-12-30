function [ modelOut ] = PLUS( model, grad, step )
% Add 2 structs. Can be used to add the gradients to the model


modelOut.W1 = model.W1 - step(1)*grad.W1;
modelOut.b1 = model.b1 - step(2)*grad.b1;

modelOut.W2 = model.W2 - step(3)*grad.W2;
modelOut.b2 = model.b2 - step(4)*grad.b2;

modelOut.W3 = model.W3 - step(5)*grad.W3;
modelOut.b3 = model.b3 - step(6)*grad.b3;

modelOut.W4 = model.W4 - step(7)*grad.W4;
modelOut.b4 = model.b4 - step(8)*grad.b4;




end

