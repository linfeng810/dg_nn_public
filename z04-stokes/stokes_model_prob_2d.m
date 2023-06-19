% this is to calculate rhs of 2d stokes manufactured solution
% -nabla \cdot (\nabla u) + \nabla p = f
% velocity should satisfy divergence-free condition
% \nabla \cdot u = 0

clear;

syms x y z
pi = sym(pi);

u = sym(zeros(2,1));

% PROBLEM Cockburn, Kanschat ...
u(1) = -exp(x)*(y*cos(y)+sin(y));
u(2) =exp(x)*y*sin(y);
p = 2*exp(x)*sin(y);

gradu = sym(zeros(2,2));
for i = 1:2
    gradu(:,i) = gradient(u(i), [x y ]);
end
gradTu = transpose(gradu);

divu = divergence(u, [x y ]);
gradp = gradient(p, [x y ]);

sigma = gradu + gradTu;  % * mu (viscosity) we omit mu since mu=1

f = sym(zeros(2,1));
for i = 1:2
    f(i) = -divergence(sigma(i,:), [x y ]) + gradp(i);
end
