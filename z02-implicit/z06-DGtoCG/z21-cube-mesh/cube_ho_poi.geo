//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Characteristic Length {4, 3, 1, 2, 5, 6, 7, 8} = 2;


//+
Physical Surface("diri") = {1, 4, 3, 2, 5};
//+
Physical Surface("neu") = {6};  // this is z=1 plane
//+
// Physical Surface("symmetry") = {6, 5};
//+
Physical Volume("fluid") = {1};
