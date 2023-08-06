% this is to compute 3D NS model problem sln
% rhs and proper bc

syms x y z t
u = sym(zeros(3,1));

% PROBLEM BELTRAMI FLOW (from Liu 2019)
rho = 1;
mu = 1;
u(1) = -exp(-t+x)*sin(y+z) - exp(-t+z)*cos(x+y);
u(2) = -exp(-t+y)*sin(x+z) - exp(-t+x)*cos(y+z);
u(3) = -exp(-t+z)*sin(x+y) - exp(-t+y)*cos(x+z);
p = -exp(-2*t) * (exp(x+z)*sin(y+z)*cos(x+y) ...
    +exp(x+y)*sin(x+z)*cos(y+z) ...
    +exp(y+z)*sin(x+y)*cos(x+z) ...
    +1/2*exp(2*x)+1/2*exp(2*y)+1/2*exp(2*z)); % -7.639581710561036);

% % PROBLEM poiseuille
% u(1) = 1 - y^2;
% u(2) = 0;
% u(3) = 0;
% p = -2*x+2;

gradu = sym(zeros(3,3));
for i = 1:3
    gradu(i,:) = gradient(u(i), [x y z]);
end
gradTu = transpose(gradu);

divu = divergence(u, [x y z]);
gradp = gradient(p, [x y z]);
sigma = gradu + gradTu - p*eye(3,3);  % real stress
% sigma = gradu - p*eye(3,3);  % pseudo stress

unablau = sym(zeros(3,1));
for i = 1:3
    unablau(i) = u(1)*diff(u(i), x) + u(2) * diff(u(i), y) ...
        + u(3)*diff(u(i), z);
end

f = sym(zeros(3,1));
for i = 1:3
    f(i) = unablau(i) - divergence(sigma(i,:), [x y z]);
end

% neumann bc (z=1)
bc_neu_z1 = subs(gradu-p*eye(3,3), z, 1) * [0, 0, 1]'