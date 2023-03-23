% first read in x_all and sfc_no
x_all = readmatrix('x_all.txt');
% sfc_no = readmatrix('sfc.txt');
% sfc_no = readmatrix('whichd.txt');
sfc_no = readmatrix('inv_no.txt')+1;
% [out, idx] = sort(sfc_no);
% manually define sfc_no
% sfc_no=[1,2,20,19, 22,21,23,4,...
%     3,15,14,17, 16,18,5,6,...
%     28,26,27,24, 25,8,7,13,...
%     11,12,9,10, 41,39,40,57,...
%     33,34,36,38, 63,51,52,54, ...
%     56,64,50,47, 45,46,61,44,...
%     43,42,58,35, 37,59,53,55,...
%     62,49,48,60, 31,29,30,32];

% sfc_no=[ 2  3 14 13 16 35 37 33 ...
%     22 20 21 18 56 39 41 38 ...
%     19  1 32 34 36 58 29 31 ...
%      28 52 54 50 57 62 15 17 ...
%      4  5 51 53  0  9  8 11 ...
%      40 43 10 12  6  7 44 46 ...
%      45 60 42 59 30 47 48 24 ...
%      23 49 63 26 25 27 55 61]+1;

nele = length(sfc_no);

figure(2);
clf;
axis equal
hold on
% plot mesh with ele no label
for ele=1:nele
    x1 = x_all((ele-1)*10+1,:);
    x2 = x_all((ele-1)*10+2,:);
    x3 = x_all((ele-1)*10+3,:);
    x10 = x_all((ele-1)*10+10,:);
    plot([x1(1),x2(1)], [x1(2),x2(2)],'k-');
    plot([x2(1),x3(1)], [x2(2),x3(2)],'k-');
    plot([x3(1),x1(1)], [x3(2),x1(2)],'k-');
    text(x10(1),x10(2),num2str(ele), 'FontSize', 10);
    ele_center(ele,1:2) = x10;
end
xlim([0,1])
ylim([0,1])
axis off

% plot sfc
% plot(ele_center(sfc_no,1), ele_center(sfc_no,2), 'b-');

for ele=1:nele-1
    plot(ele_center(sfc_no(ele:ele+1),1), ele_center(sfc_no(ele:ele+1),2),'b-');
    pause(0.1);
end