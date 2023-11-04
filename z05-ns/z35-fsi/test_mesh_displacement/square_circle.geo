sz1 = 0.1;
sz2 = 0.02;
Point(1) = {0,0,0,sz1};
Point(2) = {1,0,0,sz1};
Point(3) = {1,1,0,sz1};
Point(4) = {0,1,0,sz1};
Point(5) = {0.25, 0.25, 0, sz2};
//+
Point(6) = {0.2, 0.25, 0, sz2};
//+
Point(7) = {0.25, 0.3, 0, sz2};
//+
Point(8) = {0.3, 0.25, 0, sz2};
//+
Point(9) = {0.25, 0.2, 0, sz2};
//+
Circle(5) = {6, 5, 7};
//+
Circle(6) = {7, 5, 8};
//+
Circle(7) = {8, 5, 9};
//+
Circle(8) = {9, 5, 6};
//+
Line(9) = {1, 2};
//+
Line(10) = {2, 3};
//+
Line(11) = {3, 4};
//+
Line(12) = {4, 1};
//+
Curve Loop(1) = {12, 9, 10, 11};
//+
Curve Loop(2) = {5, 6, 7, 8};
//+
Plane Surface(1) = {1, 2};
//+
Plane Surface(2) = {2};
//+
Physical Curve("diri_f", 13) = {12, 11, 9};
//+
Physical Curve("neu_f", 14) = {10};
//+
Physical Curve("diri_s", 15) = {5, 8};
//+
Physical Curve("neu_s", 16) = {6, 7};
//+
Physical Surface("fluid", 17) = {1};
//+
Physical Surface("solid", 18) = {2};
