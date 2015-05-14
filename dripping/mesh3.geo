Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, -0, 0, 1.0};
Delete {
  Point{1};
}
Delete {
  Point{2};
}

dens = 0.1;

Point(2) = {-0, -0, 0, dens};
Point(3) = {1, 0, -0, dens};
Point(4) = {1, 2, -0, dens};
Point(5) = {0.6, 2, -0, dens};
Point(6) = {0.4, 2, -0, dens};
Point(7) = {0, 2, -0, dens};
Point(8) = {0.4, 2.4, -0, dens};
Point(9) = {0.6, 2.4, -0, dens};
Line(1) = {7, 2};
Line(2) = {3, 2};
Line(3) = {6, 7};
Line(4) = {8, 6};
Line(5) = {9, 5};
Line(6) = {4, 5};
Line(7) = {3, 4};
Line(8) = {8, 9};
Line Loop(9) = {3, 1, -2, 7, 6, -5, -8, 4};
Plane Surface(10) = {9};

Mesh.SubdivisionAlgorithm=3;
Mesh.Algorithm=8
