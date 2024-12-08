// Parameters
L = 10;      // Length of the plate
H = 10;       // Height of the plate
r = 3;       // Radius of the hole
lc = 0.2;    // Mesh size parameter

// Define the plate geometry
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, H, 0, lc};
Point(4) = {0, H, 0, lc};

// Define the hole geometry
Point(5) = {L/2, H/2, 0, lc};          // Center of the hole
Point(6) = {L/2 + r, H/2, 0, lc};      // Start point on the circle
Point(7) = {L/2, H/2 + r, 0, lc};      // Top point on the circle
Point(8) = {L/2 - r, H/2, 0, lc};      // Left point on the circle
Point(9) = {L/2, H/2 - r, 0, lc};      // Bottom point on the circle

// Plate lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Define circular arcs for the hole
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// Create surface with the hole
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};

// Define physical groups for boundary conditions
Physical Curve("PlateRightLeft") = {2, 4};
Physical Curve("PlateExterior") = {1, 2, 3, 4};
Physical Curve("HoleInterior") = {5, 6, 7, 8};
Physical Surface("PlateWithHole") = {1};

// Mesh the geometry
Mesh 2;
