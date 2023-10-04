//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 0.5, 0, 1.0};
//+
Point(4) = {0.5, 0.5, 0, 1.0};
//+
Point(5) = {0.5, 1, 0, 1.0};
//+
Point(6) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 1};
//+
Line(7) = {1, 4};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6};
//+
Curve Loop(2) = {3, -7, 1, 2};
//+
Plane Surface(1) = {2};
//+
Curve Loop(3) = {4, 5, 6, 7};
//+
Plane Surface(2) = {3};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Physical Curve("top", 8) = {5};
//+
Physical Curve("left", 9) = {6};
//+
Physical Curve("bottom", 10) = {1};
//+
Physical Curve("right", 11) = {2};
//+
Physical Surface("domain", 12) = {2, 1};
