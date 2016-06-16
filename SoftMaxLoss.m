function [ L ] = SoftMaxLoss( u, y )

% p_j = -log(  \frac{ e^{u_j} }{  \sum_k e^{u_k}  }  )
% L = dot( p, y )

epsilon = 1E-8;

% u and y must be of same size
exp_u = exp( u );
sigma = sum( exp_u + epsilon );

p = -log( exp_u + epsilon / sigma );

L = dot( p, y );

end

