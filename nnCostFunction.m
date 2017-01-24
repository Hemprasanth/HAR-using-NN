function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=sigmoid(z2);
k=size(z2,1);
a2=[ones(k,1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);

u1=y_matrix;
u2=log(a3);
u3=1-y_matrix;
u4=log(1-a3);

v1=u1.*u2;
v2=u3.*u4;

w1=sum(v1,1);
w2=sum(w1);
w3=sum(v2,1);
w4=sum(w3);



Teta1=Theta1(:,2:end).^2;
Teta2=Theta2(:,2:end).^2;
J=(-1/m)*(w2+w4)+(lambda/(2*m))*(sum(sum(Teta1))+sum(sum(Teta2)));

d3=a3-y_matrix;
d2=(d3*Theta2(:,2:end)).*(sigmoidGradient(z2));
Delta1=d2'*a1;
Delta2=d3'*a2;
t1=Theta1;
t1(:,1)=0;
t2=Theta2;
t2(:,1)=0;
Theta1_grad=((1/m)*Delta1)+(lambda/m)*t1;
Theta2_grad=((1/m)*Delta2)+(lambda/m)*t2;














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
