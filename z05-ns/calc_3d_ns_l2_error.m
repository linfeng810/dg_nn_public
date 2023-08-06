% calculate l2 norm of error in 3d stokes model problem

% clear;

ux_all = readmatrix('x_all.txt');
u_allt = readmatrix('u_all.txt');
p_allt = readmatrix('p_all.txt');
px_all = readmatrix('p_x_all.txt');
u_nloc = 20;
p_nloc = 10;
u_nonods = size(ux_all,1);
p_nonods = size(px_all,1);
nele = u_nonods/u_nloc;
ndim = 3;
sq_sum = 0;
l_inf = 0;

% u_all reshape:
u_all = u_allt(2,:);
u_all = reshape(u_all, ndim, u_nloc*nele);
u_ana = zeros(ndim, u_nonods);
% p_all 
p_all = p_allt(2,:);  % p_nloc*nele
p_ana = zeros(p_nonods, 1);

% get u error
for i = 1:u_nonods
    uxi = u_all(1,i);
    uyi = u_all(2,i);
    uzi = u_all(3,i);

    xi = ux_all(i,1);
    yi = ux_all(i,2);
    zi = ux_all(i,3);

    u_ana(1, i) = -2/3*sin(xi)^3;
    u_ana(2, i) = sin(xi)^2 * (yi*cos(xi) - zi*sin(xi));
    u_ana(3, i) = sin(xi)^2 * (zi*cos(xi) + yi*sin(xi));

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

fprintf('%d elements appximation, velocity error:\n', nele);
l2norm = sqrt(sq_sum)/u_nonods/3;
fprintf('l2 norm: %.4e\n', l2norm);
fprintf('l_inf norm: %.4e\n', l_inf);

figure(2);clf;
plot(reshape(u_ana, ndim*u_nonods, 1), 'x'); hold on; plot(reshape(u_all, ndim*u_nonods, 1), 'o')
legend('analytical', 'numerical')
title({[int2str(nele), ' element approximation'],...
    ['v l2/linf norm ', num2str(l2norm), '/', num2str(l_inf)]})

% get p error
sq_sum = 0;
l_inf = 0;
for i = 1:p_nonods
    pi = p_all(i);

    xi = px_all(i,1);
    yi = px_all(i,2);
    zi = px_all(i,3);

    p_ana(i) = sin(xi);
%     p_ana(i) = xi;

%     l_inf = max(l_inf, abs(p_ana(i) - pi) );
%     sq_sum = sq_sum ...
%         +(p_ana(i) - pi)^2;
%     disp(sq_sum)
end

% % remove average of pressure then compare
p_num_ave = sum(p_all) / p_nonods
p_ana_ave = sum(p_ana) / p_nonods
p_ave_diff = p_ana_ave - p_num_ave
p_all = p_all + p_ave_diff;
% p_ana = p_ana - sum(p_ana) / p_nonods;
% now compute error
l_inf = max(abs(p_ana - p_all'));
l2norm = norm(p_ana - p_all')/p_nonods;

fprintf('pressure error:\n');

fprintf('l2 norm: %.4e\n', l2norm);
fprintf('l_inf norm: %.4e\n', l_inf);

figure(3);clf;
plot(p_ana, 'x'); hold on; plot(p_all, 'o')
legend('analytical', 'numerical')
title({[int2str(nele), ' element approximation'],...
    ['p l2/linf norm ', num2str(l2norm), '/', num2str(l_inf)]})