// this is the geometry used in FSI Benchmark often
// addressed as Turek & Hron benchmark
// a soft band located behind a cylinder, compliant
// of votex shedding

// Reference: TUREK, S., HRON, J., RAZZAQ, M., WOBKER,
//  H. & SCHÄFER, M. Numerical Benchmarking of Fluid-
//  Structure Interaction: A Comparison of Different 
//  Discretization and Solution Approaches. 2010 Berlin, 
//  Heidelberg. Springer Berlin Heidelberg, 413-424.

// Parameter
L = 2.5;
H = 0.41;
C = 0.2;
r = 0.05;
l = 0.35;
h = 0.02;
b = Sqrt(r^2 - (h/2)^2);

// mesh size
sz1 = H/10;
sz2 = h/2;

Point(1) = {0, 0, 0, sz1};
Point(2) = {L, 0, 0, sz1};
Point(3) = {L, H, 0, sz1};
Point(4) = {0, H, 0, sz1};

Point(5) = {C, C, 0, sz2};
Point(6) = {C+b, C-h/2, 0, sz2};
Point(7) = {C+b, C+h/2, 0, sz2};
Point(8) = {C, C+r, 0, sz2};
Point(9) = {C-r, C, 0, sz2};
Point(10)= {C, C-r, 0, sz2};

Point(11)= {C+r+l, C-h/2, 0, sz2};
Point(12)= {C+r+l, C+h/2, 0, sz2};

Line(1) = {1,2};
Line(2) = {1,4};
Line(3) = {2,3};
Line(4) = {4,3};

Circle(5) = {6,5,7};
Circle(6) = {7,5,8};
Circle(7) = {8,5,9};
Circle(8) = {9,5,10};
Circle(9) = {10,5,6};

Line(10) = {6,11};
Line(11) = {11,12};
Line(12) = {12, 7};

Line Loop(13) = {1,3,-4, -2};
Line Loop(14) = {6, 7, 8, 9, 10, 11, 12};
Line Loop(15) = {-5, 10, 11, 12};

Plane Surface(1) = {13, 14};
Plane Surface(2) = {15};

// old boundary marks for IC-FERST
// Physical Line(1) = {2}; // inlet
// Physical Line(2) = {3}; // outlet
// Physical Line(3) = {1, 4}; // wall
// Physical Line(4) = {5,6,7,8,9}; //cylinder wall
// // rest of the lines are solid-fluid interface
// Physical Surface(1) = {1}; // fluid
// Physical Surface(2) = {2}; // solid

// new boundary marks for AI4CFD code
//+
Physical Curve("diri_f", 16) = {2, 4, 1, 9, 8, 7, 6};
//+
Physical Curve("neu_f", 17) = {3};
//+
Physical Curve("diri_s", 18) = {5};

//+
Physical Surface("fluid") = {1};
//+
Physical Surface("solid") = {2};
