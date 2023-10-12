//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, 0, 0, 1.0};
//+
Point(5) = {0, 2, 0, 1.0};
//+
Point(6) = {1, 2, 0, 1.0};
//+
Line(1) = {1, 4};
//+
Line(2) = {4, 3};
//+
Line(3) = {3, 2};
//+
Line(4) = {2, 1};
//+
Line(5) = {2, 5};
//+
Line(6) = {5, 6};
//+
Line(7) = {6, 3};
//+
Curve Loop(1) = {4, 1, 2, 3};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 6, 7, 3};
//+
Plane Surface(2) = {2};
//+
Physical Curve("diri_f") = {4, 2};
//+
Physical Curve("neu_f") = {1};
//+
Physical Curve("diri_s") = {5, 7};
//+
Physical Curve("neu_s") = {6};
//+
Physical Curve("interface") = {3};
Physical Surface("fluid") = {1};
Physical Surface("solid") = {2};