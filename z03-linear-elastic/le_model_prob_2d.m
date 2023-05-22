% this is to calculate the rhs of 3D linear elastic model problem
% - \nabla . (\lambda \nabla . u + \mu (\nabla . u + \nabla^T . u)) = f
% 
% analytical solution:
% u_x = pi * cos(pi*y) sin^2(pi*x) * sin(pi*y)
% u_y = -pi* cos(pi*x) sin(pi*x) sin^2(pi*y)

clear;
syms x y
a = 1;
u = sym(zeros(2,1));
u(1) = pi * cos(pi * y) * sin(pi * x)^2 * sin(pi * y);
u(2) = - pi * cos(pi * x) * sin(pi * x) * sin(pi * y)^2;

gradu = sym(zeros(2,2));
for i = 1:2
    gradu(:,i) = gradient(u(i), [x y]);
end
gradTu = transpose(gradu);
divuI = divergence(u, [x y])*eye(2,2);

lambda = 1;
mu = 1;
f = sym(zeros(2,1));
sigma = lambda * divuI + mu * (gradu + gradTu);

for i = 1:2
    f(i) = -divergence(sigma(:,i), [x y]);
end