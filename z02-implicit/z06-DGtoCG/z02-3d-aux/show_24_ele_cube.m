x_all_cg = readmatrix('../x_all_cg.txt');
x_all_dg = readmatrix('../x_all.txt');
cg_ndglno = readmatrix('../cg_ndglno.txt');
cg_ndglno = reshape(cg_ndglno, [4, 24])+1;
nele = 24;
fig = figure(1);
clf; 
ax = axes;
xlabel('x')
ylabel('y')
zlabel('z')
hold on;
axis equal
ele_to_plt = 5;
% for ele = 1:nele
% for ele = colele(finele(ele_to_plt)+1:finele(ele_to_plt+1))+1
for ele = [0,  1,  2,  4,  5,  7,  8, 10, 12, 13, 15, 17]+1
    plab = {int2str((ele-1)*4+1), ...
        int2str((ele-1)*4+2),...
        int2str((ele-1)*4+3),...
        int2str((ele-1)*4+4)};
    flab = plab;
    vlab = int2str(ele);
    plot_tetra(ax, x_all_cg(cg_ndglno(:,ele)', :), ...
        plab, flab, vlab, 0.75,...
        x_all_dg((ele-1)*20+1:ele*20, :))
end