// FDA Nozzle benchmark
// c.f. Fehn et al. Int J Numer Meth Biomed Engng 2019; 35; e3228
d = 0.004;
D = 3 * d;
L_i = 4 * D;
L_c = (D - d) / 0.3455848697103184;  // 2arctan(pi/18) is on denominator
L_th = 10 * d;
L_o = 10 * D;

sz1 = d/4;
sz2 = D/4;

// Point(1) = {0,0,0,sz1};
// Point(2) = {0,0,L_o, sz2};
// Point(3) = {0,0,-L_th, sz1};
// Point(4) = {0, 0, -L_th - L_c, sz1};
// Point(5) = {0, 0, -L_th - L_c - L_i, sz1};

// Point(6) = {D/2,0,0,sz1};
// Point(7) = {D/2,0,L_o, sz2};
// Point(8) = {d/2,0,-L_th, sz1};
// Point(9) = {D/2, 0, -L_th - L_c, sz1};
// Point(10) = {D/2, 0, -L_th - L_c - L_i, sz1};
// Point(11) = {d/2, 0, 0, sz1};

// Point(12) = {-D/2,0,0,sz1};
// Point(13) = {-D/2,0,L_o, sz2};
// Point(14) = {-d/2,0,-L_th, sz1};
// Point(15) = {-D/2, 0, -L_th - L_c, sz1};
// Point(16) = {-D/2, 0, -L_th - L_c - L_i, sz1};
// Point(17) = {-d/2, 0, 0, sz1};//+
SetFactory("OpenCASCADE");
Cylinder(1) = {0, 0, 0, 0, 0, L_o, D/2, 2*Pi};
Cylinder(2) = {0, 0, 0, 0, 0, -L_th, d/2, 2*Pi};
Cone(3) = {0, 0, -L_th, 0, 0, -L_c, d/2, D/2, 2*Pi};
Cylinder(4) = {0, 0, -L_th - L_c, 0, 0, -L_i, D/2, 2*Pi};//+
BooleanUnion{ Volume{1}; Delete; }{ Volume{2}; Volume{3}; Volume{4}; Delete; }
//+
Dilate {{0, 0, 0}, {1., 1., 1}} {
  Point{1}; Point{2}; Point{3}; Point{4}; Point{5}; Point{6}; Curve{1}; Curve{2}; Curve{4}; Curve{3}; Curve{6}; Curve{5}; Curve{8}; Curve{7}; Curve{10}; Curve{9}; Surface{2}; Surface{3}; Surface{1}; Surface{4}; Surface{5}; Surface{6}; Surface{7}; Volume{1}; 
}

//+
MeshSize {2, 3, 4} = sz1;
//+
MeshSize {1, 5, 6} = sz2;
//+
Physical Surface("diri", 11) = {7, 2};
//+
Physical Surface("neu", 12) = {1, 3, 4, 5, 6};
//+
Physical Volume("fluid", 14) = {1};
