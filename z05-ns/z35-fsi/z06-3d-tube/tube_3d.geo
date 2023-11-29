SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 0.5, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 0.6, 0, 2*Pi};
//+
Curve Loop(1) = {1};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2};
//+
Plane Surface(2) = {2};
//+
BooleanDifference{ Surface{2}; Delete; }{ Surface{1}; Delete; }
//+
Curve Loop(1) = {2};
//+
Plane Surface(3) = {1};
//+
Extrude {0, 0, 5} {
  Surface{2}; 
}
//+
Extrude {0, 0, 5} {
  Surface{3}; 
}
//+
Physical Surface("diri_f", 9) = {8};
//+
Physical Surface("neu_f", 10) = {3};
//+
Physical Surface("diri_s", 11) = {2, 6};
//+
Physical Surface("neu_s", 12) = {4};
//+
Physical Volume("fluid", 13) = {2};
//+
Physical Volume("solid", 14) = {1};
//+
MeshSize {2, 1, 5, 4, 3} = 0.1;
