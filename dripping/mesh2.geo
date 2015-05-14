Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0.4, 1, 0, 1.0};
Point(5) = {0.6, 1, 0, 1.0};
Point(6) = {0.6, 1.4, 0, 1.0};
Point(7) = {0.4, 1.4, 0, 1.0};
Point(8) = {1, 1.4, 0, 1.0};
Delete {
  Point{8};
}
Point(8) = {-0, 1, 0, 1.0};
Line(1) = {8, 4};
Line(2) = {7, 4};
Line(3) = {6, 7};
Line(4) = {6, 5};
Line(5) = {5, 3};
Line(6) = {3, 2};
Line(7) = {2, 1};
Line(8) = {1, 8};
Line Loop(9) = {1, -2, -3, 4, 5, 6, 7, 8};
Plane Surface(10) = {9};
Translate {0, 0, 1} {
  Point{7, 6, 4, 8, 3, 5};
}
Translate {0, 1, 1} {
  Point{7, 6, 4, 8, 5, 3};
}
Translate {0, 0, -1} {
  Point{6, 7};
}
