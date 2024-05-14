h = 0.005;
sz1 = 2*h;
sz2 = 0.1;
sz3 = 0.05;

Point(1) = {0, 0, 0, sz2};
Point(2) = {0, -.5, 0, sz2};
Point(3) = {0.495, -.5, 0, sz1};
Point(4) = {0.5, -.5, 0, sz1};
Point(5) = {1.25, -0.2, 0, sz3};
Point(6) = {1.75, -0.2, 0, sz3};
Point(7) = {1.75, 0.2, 0, sz3};
Point(8) = {1.25, 0.2, 0, sz3};
Point(9) = {0.5, 0.5, 0, sz2};
Point(10) = {0, 0.5, 0, sz2};
Point(11) = {0.495, -0.25, 0, sz1};
Point(12) = {0.5, -0.25, 0, sz1};
Line(1) = {10, 1};
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 11};
Line(5) = {11, 12};
Line(6) = {12, 4};
Line(7) = {4, 3};
Line(8) = {4, 5};
Line(9) = {5, 6};
Line(10) = {6, 7};
Line(11) = {7, 8};
Line(12) = {8, 9};
Line(13) = {9, 10};
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13};
Plane Surface(1) = {1};
Curve Loop(2) = {4, 5, 6, 7};
Plane Surface(2) = {2};
Physical Curve("diri_f", 14) = {1, 2, 3, 8, 9, 13, 12, 11};
Physical Curve("neu_f", 15) = {10};
Physical Curve("diri_s", 16) = {7};
//+
Physical Surface("fluid", 17) = {1};
//+
Physical Surface("solid", 18) = {2};