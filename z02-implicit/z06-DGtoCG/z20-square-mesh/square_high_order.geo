sz=2.0;

//+
Point(1) = {0, 0, 0, sz};
//+
Point(2) = {0, 1, 0, sz};
//+
Point(3) = {1, 1, 0, sz};
//+
Point(4) = {1, 0, 0, sz};
//+
Line(1) = {1, 4};
//+
Line(2) = {2, 3};
//+
Line(3) = {1, 2};
//+
Line(4) = {4, 3};
//+
Line Loop(1) = {3, 2, -4, -1};
//+
Plane Surface(1) = {1};
//+
Physical Curve("diri_f", 5) = {3, 4, 1, 2};
// Physical Curve("neu_f", 6) = {2};
//+
Physical Surface("fluid", 7) = {1};
