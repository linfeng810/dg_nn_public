% calc l2 norm of error in 2D stokes model problem

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
ndim = 2;
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

    xi = ux_all(i,1);
    yi = ux_all(i,2);

%     % PROBLEM: poiseuille
%     u_ana(1,i) = yi-yi^2;
%     u_ana(2,i) = 0;

%     % PROBLEM: ns (lid-driven)
%     u_ana(1,i) = 1/4*(xi-2)^2*xi^2*yi*(yi^2-2);
%     u_ana(2,i) = -1/4*xi*(xi^2-3*xi+2)*yi^2*(yi^2-4);

    % problem: kovasznay
     Reyno = 1;
    lamb = Reyno/2-sqrt(Reyno^2/4+4*pi^2);
    u_ana(1,i) = 1 - exp(lamb*xi)*cos(2*pi*yi);
    u_ana(2,i) = lamb/2/pi*exp(lamb*xi)*sin(2*pi*yi);

    l_inf = max(l_inf, sqrt(...
        + (uxi-u_ana(1,i))^2 ...
        + (uyi-u_ana(2,i))^2) );
    sq_sum = sq_sum ...
        + (uxi-u_ana(1,i))^2 ...
        + (uyi-u_ana(2,i))^2;
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
    
%     % PROBLEM poiseuille
%     p_ana(i) = -2*xi+1;

%     % PROBLEM ns lid-driven
%     Reyno = 1;
%     p_ana(i) = (xi*xi*(3*xi^4-15*xi^3+10*xi^2*xi^2-30*xi*(xi^2-2) ...
%         +20*(xi^2-2)))/5/Reyno ...
%     - 1/128*(xi-2)^4*xi^4*xi^2*(xi^4-2*xi^2+8) ...
%     - (8/(5*Reyno) - 1408/33075);

    % problem Kovasznay
    Reyno = 1;
    lamb = Reyno/2-sqrt(Reyno^2/4+4*pi^2);
    p_ana(i) = -0.5*exp(2*lamb*xi);
end

% remove average of pressure then compare
p_num_ave = sum(p_all) / p_nonods
p_ana_ave = sum(p_ana) / p_nonods
p_ave_diff = p_ana_ave - p_num_ave
p_all = p_all + p_ave_diff;
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