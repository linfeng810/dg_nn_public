
h = 0.05;
L = 5;

sz1 = h/2;

Point(1) = {0,0,0,sz1};
Point(2) = {L,0,0,sz1};
Point(3) = {L,h,0,sz1};
Point(4) = {0,h,0,sz1};
//+
Line(1) = {1, 4};
//+
Line(2) = {4, 3};
//+
Line(3) = {3, 2};
//+
Line(4) = {2, 1};
//+
Physical Curve("diri", 5) = {1, 3};
//+
Physical Curve("neu", 6) = {4, 2};
//+
Curve Loop(1) = {2, 3, 4, 1};
//+
Plane Surface(1) = {1};
//+
Physical Surface("fluid", 7) = {1};
