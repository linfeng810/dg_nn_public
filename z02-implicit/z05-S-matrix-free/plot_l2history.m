nele = 1024;

l2history_mg = readmatrix(['r0l2all_',num2str(nele),'_mg.txt']);
l2history_jac = readmatrix(['r0l2all_',num2str(nele),'_jacobi.txt']);

figure(1); clf;
len = size(l2history_mg,1);
step = floor(len/300);
semilogy(1:step:len, l2history_mg(1:step:len), LineWidth=2);
hold on
len = size(l2history_jac,1);
step = floor(len/300);
semilogy(1:step:len, l2history_jac(1:step:len), LineWidth=2);

xlabel('MG cycles / Jacobi iterations');
ylabel('L2 of residuals');
title(['num of elements: ', num2str(nele)])
legend('Multigrid cycle', 'Plain Jacobian iteration')
grid on