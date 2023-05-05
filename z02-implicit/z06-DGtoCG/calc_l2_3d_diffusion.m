% calculate l2 norm of error in 3d diffusion model problem
% problem statement:
% -\nabla^2 c = f
% analytical solution: 
%  sin(2 pi x) sin(2 pi y) sin(2 pi z)
% rhs source term:
%  8 pi^2 sin(2 pi x) sin(2 pi y) sin(2 pi z)

x_all = readmatrix('x_all.txt');
c_all = readmatrix('c_all.txt');
nloc = 20;
nele = size(c_all,2)/nloc;

nonods = size(x_all,1);
sq_sum = 0;
l_inf = 0;

c_ana = zeros(nonods,1);

for i = 1:nonods
    ai = c_all(2,i);
    xi = x_all(i,1);
    yi = x_all(i,2);
    zi = x_all(i,3);
    bi = sin(2*pi*xi)*sin(2*pi*yi)*sin(2*pi*zi);
%     bi = yi;
    c_ana(i) = bi;
%     ai-bi
    l_inf = max(l_inf, abs(ai-bi));
    sq_sum = sq_sum + (ai-bi)^2;
end

nele
l2norm = sqrt(sq_sum)/nonods
l_inf

figure();clf;
plot(c_ana, 'x'); hold on; plot(c_all(2,:), 'o')
legend('analytical', 'numerical')
title({[int2str(nele), ' element approximation'],...
    ['l2/linf norm ', num2str(l2norm), '/', num2str(l_inf)]})