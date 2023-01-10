sz=2.0;

//+
Point(1) = {0, 0, 0, sz};
//+
Point(2) = {1, 0, 0, sz};
//+
Point(3) = {1, 1, 0, sz};
//+
// Point(4) = {1, 0, 0, sz};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 1};
//+
// Line(4) = {4, 3};
//+
Line Loop(1) = {1,2,3};
//+
Plane Surface(1) = {1};
