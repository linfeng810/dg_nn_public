% calculate jacobian

% get shape functions
% nlxyzgi;

% local nodes coordinates (on reference ele)
x_loc=[1.0,0; 0,1.0; 0,0;...
    2/3,1/3;1/3,2/3;0,2/3;0,1/3;1/3,0;2/3,0;
    1/3,1/3];
x_loc=x_loc';
% # first we calculate jacobian matrix (J^T) = [j11,j12;
% #                                             j21,j22]
% # [ d x/d xi,   dy/d xi ;
% #   d x/d eta,  dy/d eta]
j11=squeeze(nlx(:,1,:))*x_loc(1,:)'; % should be 1
j12=squeeze(nlx(:,1,:))*x_loc(2,:)'; % should be 0
j21=squeeze(nlx(:,2,:))*x_loc(1,:)'; % should be 0
j22=squeeze(nlx(:,2,:))*x_loc(2,:)'; % should be 1
% squeeze(nlx(:,1,:))
% x_loc(1,:)'
% x_loc(2,:)'

det = j11.*j22-j12.*j21;

invj11 = j11./det;
invj12 = -j21./det;
invj21 = -j12./det;
invj22 = j22./det;

squeeze(nlx(:,2,:))