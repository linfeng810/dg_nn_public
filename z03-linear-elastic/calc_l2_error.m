ndim=2;
% first read in u_all and x_all

% u_all: (n_timestep , ndim*nonods)
% x_all: (nonods, 2)
u_allt = readmatrix('u_all.txt');
x_all = readmatrix('x_all.txt');
% l2 norm of a-b
% sqrt( (a1-b1)^2 + (a2-b2)^2 + ... )

nonods = size(x_all,1);
nele = nonods/10;
sq_sum = 0 ; % sum of squares
l_inf = 0;

% u_all reshape:
u_all1 = u_allt(2,:);
u_all = [u_all1(1:nele*10); u_all1(nele*10+1:end)];

u_ana = zeros(ndim, nonods); % analytical solution

for i = 1:nonods 
    uxi = u_all(1,i);
    uyi = u_all(2,i);
    xi = x_all(i,1);
    yi = x_all(i,2);
    bxi = pi*cos(pi*yi)*sin(pi*xi)^2*sin(pi*yi);
    byi = -pi*cos(pi*xi)*sin(pi*xi)*sin(pi*yi)^2;
%     bxi = sin(pi*xi)*sinh(pi*yi)/sinh(pi);
%     byi = sin(pi*xi)*sinh(pi*yi)/sinh(pi);
    u_ana(1,i) = bxi;
    u_ana(2,i) = byi;
    l_inf = max(l_inf, sqrt((uxi-bxi)^2+(uyi-byi)^2) );
    sq_sum = sq_sum + (uxi-bxi)^2+(uyi-byi)^2;
end

l2norm = sqrt(sq_sum)/nonods/2
l_inf

bc4_idx = x_all(:,2)==1;

figure(1); clf;
ax1 = subplot(1,2,1);
plot3(ax1, ...
    x_all(:,1), x_all(:,2), u_all(1,:),'x', ...
    x_all(:,1), x_all(:,2), u_ana(1,:),'o');
xlabel('x')
ylabel('y')
zlabel('u_x')
ax2 = subplot(1,2,2);
plot3(ax2, ...
    x_all(:,1), x_all(:,2), u_all(2,:),'x', ...
    x_all(:,1), x_all(:,2), u_ana(2,:),'o');
xlabel('x')
ylabel('y')
zlabel('u_y')

% add title
sgtitle([num2str(nele), ' elements approximation']);
fig = gcf;
fig.Position(3) = fig.Position(3) + 250;
% add legend
Lgnd = legend('numerical', 'analytical');
Lgnd.Position(1) = 0.01;
Lgnd.Position(2) = 0.4;


% figure(2); clf;
% plot3(x_all(:,1), x_all(:,2), c_all(2,:)'-c_ana, 'x');
% xlabel('x')
% ylabel('y')
% zlabel('z : error of c')
% nele = nonods/10;
% title([num2str(nele), ' elements approximation']);
% 
% l2history = readmatrix('r0l2all.txt');
% 
% figure(3); clf;
% semilogy(l2history, LineWidth=2);
% xlabel('MG cycles');
% ylabel('L2 of residuals');
% title(['num of elements: ', num2str(nele)])