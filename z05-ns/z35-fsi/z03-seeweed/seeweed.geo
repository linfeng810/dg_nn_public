sz1 = 0.2;  // fluid
sz2 = 0.01;  // solid
//+
Point(1) = {0, 0, 0, sz1};
//+
Point(2) = {2, 0, 0, sz1};
Point(3) = {2, 0.5, 0, sz1};
Point(4) = {0, 0.5, 0, sz1};
//+
Point(6) = {0.5, 0, 0, sz2};
//+
Point(7) = {0.52, 0, 0, sz2};
//+
Point(8) = {0.52, 0.1, 0, sz2};
//+
Point(9) = {0.5, 0.1, 0, sz2};
//+
Line(1) = {1, 6};
//+
Line(2) = {6, 9};
//+
Line(3) = {9, 8};
//+
Line(4) = {8, 7};
//+
Line(5) = {7, 6};

//+
Line(7) = {4, 1};
//+
Line(8) = {4, 3};
//+
Line(9) = {3, 2};
//+
Line(10) = {2, 7};

//+
Curve Loop(1) = {2, 3, 4, -10, -9, -8, 7, 1};
//+
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, 5, 2, 3};
//+
Surface(2) = {2};
//+
Physical Curve("diri_f", 11) = {7, 8, 1, 10};
//+
Physical Curve("neu_f", 12) = {9};
//+
Physical Curve("diri_s", 13) = {5};
//+
Physical Surface("fluid", 14) = {1};
//+
Physical Surface("solid", 15) = {2};
