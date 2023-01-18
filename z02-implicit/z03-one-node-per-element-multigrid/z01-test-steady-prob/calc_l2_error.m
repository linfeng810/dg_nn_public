% first read in c_all and x_all

% c_all: (n_timestep , nonods)
% x_all: (nonods, 2)
c_all = readmatrix('c_all.txt');
x_all = readmatrix('x_all.txt');

% l2 norm of a-b
% sqrt( (a1-b1)^2 + (a2-b2)^2 + ... )

nonods = size(x_all,1);
sq_sum = 0 ; % sum of squares
l_inf = 0;

c_ana = zeros(nonods,1);

for i = 1:nonods 
    ai = c_all(2,i);
    xi = x_all(i,1);
    yi = x_all(i,2);
    bi = sin(pi*xi)*sinh(pi*yi)/sinh(pi);
    c_ana(i) = bi;
    l_inf = max(l_inf, abs(ai-bi));
    sq_sum = sq_sum + (ai-bi)^2;
end

l2norm = sqrt(sq_sum)/nonods
l_inf

bc4_idx = x_all(:,2)==1;
plot3(x_all(:,1), x_all(:,2), c_all(2,:),'x', ...
    x_all(:,1), x_all(:,2), c_ana,'o');
xlabel('x')
ylabel('y')
zlabel('z : magnitude of c')
nele = nonods/10;
title([num2str(nele), ' elements approximation']);

legend('numerical', 'analytical')