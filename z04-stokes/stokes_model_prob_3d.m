% this is to calculate rhs of 3d stokes manufactured solution
% -nabla \cdot (\nabla u) + \nabla p = f
% velocity should satisfy divergence-free condition
% \nabla \cdot u = 0

clear;

syms x y z
u = sym(zeros(3,1));

% PROBLEM Cockburn 2006
u(1) = -2/3*sin(x)^3;
u(2) = sin(x)^2 * (y*cos(x) - z*sin(x));
u(3) = sin(x)^2 * (z*cos(x) + y*sin(x));
p = sin(x);

gradu = sym(zeros(3,3));
for i = 1:3
    gradu(:,i) = gradient(u(i), [x y z]);
end
gradTu = transpose(gradu);

divu = divergence(u, [x y z]);
gradp = gradient(p, [x y z]);

sigma = gradu + gradTu;  % * mu (viscosity) we omit mu since mu=1

f = sym(zeros(3,1));
for i = 1:3
    f(i) = -divergence(sigma(i,:), [x y z]) + gradp(i);
end
