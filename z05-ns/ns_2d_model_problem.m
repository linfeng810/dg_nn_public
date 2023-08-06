% this is to compute 3D NS model problem sln
% rhs and proper bc

syms x y z t Re  
u = sym(zeros(2,1));

% problem Kovasznay
% Re = 1;
lamb = Re/2 - sqrt(Re^2/4 + 4 * pi^2);
u(1) = 1 - exp(lamb*x)*cos(2*pi*y);
u(2) = lamb / 2 / pi * exp(lamb*x) * sin(2*pi*y);
p = - 1 / 2 * exp(2*lamb*x);

% % PROBLEM from Farrell 2019
% % rho = 1, mu = 1
% u(1) = 1/4*(x-2)^2*x^2*y*(y^2-2);
% u(2) = -1/4*x*(x^2-3*x+2)*y^2*(y^2-4);
% p = (x*y*(3*x^4-15*x^3+10*x^2*y^2-30*x*(y^2-2)+20*(y^2-2)))/5/Re ...
%     - 1/128*(x-2)^4*x^4*y^2*(y^4-2*y^2+8);
% p_ave = int(int(p,x,0,2),y,0,2)/4;
% p = p - p_ave


% % PROBLEM poiseuille
% u(1) = 1 - y^2;
% u(2) = 0;
% p = -2*x+2;

gradu = sym(zeros(2,2));
for i = 1:2
    gradu(i,:) = gradient(u(i), [x y]);
end
gradTu = transpose(gradu);

divu = divergence(u, [x y]);
gradp = gradient(p, [x y]);
sigma = 1/Re*(gradu + gradTu) - p*eye(2,2);  % real stress
% sigma = gradu - p*eye(3,3);  % pseudo stress

unablau = sym(zeros(2,1));
for i = 1:2
    unablau(i) = u(1)*diff(u(i), x) + u(2) * diff(u(i), y) ;
end

f = sym(zeros(2,1));
for i = 1:2
    f(i) = unablau(i) - divergence(sigma(i,:), [x y]);
end

% neumann bc (z=1)
bc_neu_x1 = subs(1/Re*gradu-p*eye(2,2), x, 1) * [1, 0]'
bc_neu_x1_plus_adv = bc_neu_x1 - subs(u*transpose(u), x, 1) * [1, 0]'