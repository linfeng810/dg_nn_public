%% manufactured solution for fsi system
show_plot = true;
% fluid equation
% \nabla . u = 0 (continuity)
% 

% solid equation
% - \nabla . (P (F(d)) = f
% St Venant-Kirchhoff material
% d is displacement, F is deformation gradient,
% P is PK1 stress tensor

% auxiliary functions and parameters
syms x y
K = x;
M = 0.5 * x^2;
L = 1./3. * x^3;
h = 0.1;
f = 1 + h * (1-cos(2*pi*x)) * sin(2*pi*x);
a = (1 + 10 * h * cos(2*pi*x)/3)* (1-y);
b = 1.;
k = 1;
E = 2;
nu = 0.1;
syms lam_s mu_s

%% interface normal
n_inter = sym(zeros(2,1));
n_inter(1) = -diff(f, x);
n_inter(2) = 1;
n_inter_mag = sqrt(n_inter(1)^2 + n_inter(2)^2);
% normalise
n_inter = n_inter / n_inter_mag;

if show_plot  % visulise interface and normal
    figure(1);clf;
    xGrid = 0:0.05:1;
    yGrid = subs(f,x, xGrid);
    plot(xGrid, yGrid, 'x-')
    hold on
    axis equal
    nxValues = subs(n_inter(1), x, xGrid);
    nyValues = subs(n_inter(2), x, xGrid);
    quiver(xGrid, yGrid, nxValues, nyValues);
    title('interface shape and normal vector on it')
end

n_inter_f = n_inter;  % normal vec on fluid side
n_inter_s = -n_inter;  % normal vec on solid side

%% fluid velocity
u = sym(zeros(2,1));
u(1) = (k+1)*y^k*( ...
    subs(M, x, f) - subs(M, x, y)) - ...
    k*y^(k-1) * ( ...
    subs(L, x, f) - subs(L, x, y));
u(2) = y^k * (f - y) * subs(K, x, f) * diff(f, x);

if show_plot  % show velocity field
    figure(2);clf;
    [xGrid, yGrid] = meshgrid(0:0.05:1, 0:0.05:1);
    uxValues = subs(u(1), {x,y}, {xGrid, yGrid});
    uyValues = subs(u(2), {x,y}, {xGrid, yGrid});
    quiver(xGrid, yGrid, uxValues, uyValues);
    axis equal
    title('velocity field on undeformed domain')
end

gradu = sym(zeros(2,2));
gradu(1, :) = gradient(u(1), [x, y]);
gradu(2, :) = gradient(u(2), [x, y]);
A = gradu(1,1);
B = gradu(1,2);
C = gradu(2,1);
D = gradu(2,2);

%% structure displacement
d = sym(zeros(2,1));
d(1) = a;
d(2) = b * (f - 1);

if show_plot  % show deformed solid domain
    figure(3); clf;
    [xGrid, yGrid] = meshgrid(0:0.025:1, 1:0.025:1.25);
    uxValues = subs(d(1), {x, y}, {xGrid, yGrid});
    uyValues = subs(d(2), {x, y}, {xGrid, yGrid});
    plot(xGrid+uxValues, yGrid+uyValues, 'b-');
    hold on
    plot((xGrid+uxValues)', (yGrid+uyValues)', 'b-');
    axis equal
    title('deformed solid domain')
end

% deformation gradient
F = sym(zeros(2,2));
for idim =1:2
    F(idim, :) = gradient(d(idim), [x y]);
end
F = F + eye(2,2);
% Green-Lagrangian strain tensor
E = 0.5 * (transpose(F) * F - eye(2,2));
% structure stress (cauchy)
sigma_s = lam_s * trace(E) * eye(2,2) + 2 * mu_s * E;
% structure stress (PK1)
P = F * sigma_s;

% traction force at the interface
S = subs(sigma_s) * n_inter_s;

%% fluid pressure and viscosity
syms S_x  S_y  mu  p  n_x  n_y
eq1 = S_x == (mu * A - p) * n_x + mu * B * n_y;
eq2 = S_y == mu * C * n_x + (mu * D - p) * n_y;
solutions = solve([eq1, eq2], [mu, p]);

p_f = subs(solutions.p, [S_x, S_y, n_x, n_y], ...
    [S(1), S(2), n_inter_f(1), n_inter_f(2)]);
mu_f = subs(solutions.mu, [S_x, S_y, n_x, n_y], ...
    [S(1), S(2), n_inter_f(1), n_inter_f(2)]);

%% now get body force
% fluid
% \rho (u . \nabla) u - \nabla . (\mu \nabla u) + \nabla p = \rho f
% assume \rho in is 1
% gradTu = transpose(gradu);
divu = divergence(u, [x y]);
gradp = gradient(p, [x y]);
sigma = mu_f * gradu - p*eye(2,2);  % fluid pseudo stress (no gradTu term)
unablau = sym(zeros(2,1));
for i = 1:2
    unablau(i) = u(1)*diff(u(i), x) + u(2) * diff(u(i), y);
end
f_f = sym(zeros(2,1));
for i = 1:2
    f_f(i) = unablau(i) - divergence(sigma(i,:), [x y]);
end
% neumann bc (x=1)
bc_neu_x1 = subs(sigma, x, 1) * [0, 1]';

% solid
% -\nabla_X . P = \rho f
% again assum \rho is 1
% we already get PK1 stress in the above (P)
f_s = sym(zeros(2,1));
for i = 1:2
    f_s(i) = -divergence(P(i,:), [x y]);
end
