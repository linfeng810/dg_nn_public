% calculate l2 norm of error in 3d diffusion model problem
% problem statement:
% -\nabla^2 c = f
% analytical solution: 
%  sin(2 pi x) sin(2 pi y) sin(2 pi z)
% rhs source term:
%  8 pi^2 sin(2 pi x) sin(2 pi y) sin(2 pi z)

x_all = readmatrix('x_all.txt');
u_allt = readmatrix('u_all.txt');
nloc = 4;
nonods = size(x_all,1);
nele = nonods/nloc;
ndim = 3;
sq_sum = 0;
l_inf = 0;

% u_all reshape:
u_all = u_allt(2,:);
u_all = reshape(u_all, ndim, nloc*nele);
u_ana = zeros(ndim, nonods);

for i = 1:nonods
    uxi = u_all(1,i);
    uyi = u_all(2,i);
    uzi = u_all(3,i);

    xi = x_all(i,1);
    yi = x_all(i,2);
    zi = x_all(i,3);

%     u_ana(:, i) = sin(pi*(xi+1)) * sin(pi*(yi+1)) * sin(pi*(zi+1));
    u_ana(:, i) = exp(xi+yi+zi);
%     u_ana(1, i) = 0.1 * yi;
%     u_ana(2:3, i) = 0;

    l_inf = max(l_inf, sqrt(...
        + (uxi-u_ana(1,i))^2 ...
        + (uyi-u_ana(2,i))^2 ...
        + (uzi-u_ana(3,i))^2) );
    sq_sum = sq_sum ...
        + (uxi-u_ana(1,i))^2 ...
        + (uyi-u_ana(2,i))^2 ...
        + (uzi-u_ana(3,i))^2;
%     disp(sq_sum)
end

nele
l2norm = sqrt(sq_sum)/nonods/3
l_inf

figure(2);clf;
plot(reshape(u_ana, ndim*nonods, 1), 'x'); hold on; plot(reshape(u_all, ndim*nonods, 1), 'o')
legend('analytical', 'numerical')
title({[int2str(nele), ' element approximation'],...
    ['l2/linf norm ', num2str(l2norm), '/', num2str(l_inf)]})