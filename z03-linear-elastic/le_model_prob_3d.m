% this is to calculate the rhs of 3D linear elastic model problem
% - \nabla . (\lam \nabla . u + \mu (\nabla . u + \nabla^T . u)) = f
% 
% analytical solution:
% u_x = u_y = u_z = 
%     a sin(pi (x+1)) sin(pi (y+1)) sin(pi (z+1))
clear;

syms x y z
a = 1;
u = sym(zeros(3,1));
for i = 1:3
%     u(i) = a * sin(pi * (x+1)) * sin(pi * (y+1)) * sin(pi * (z+1));
    u(i) = exp(-x^2-y^2-z^2);
end

gradu = sym(zeros(3,3));
for i = 1:3
    gradu(:,i) = gradient(u(i), [x y z]);
end
gradTu = transpose(gradu);
divuI = divergence(u, [x y z])*eye(3,3);

syms lam mu
f = sym(zeros(3,1));
sigma = lam * divuI + mu * (gradu + gradTu);

for i = 1:3
    f(i) = -divergence(sigma(:,i), [x y z]);
end