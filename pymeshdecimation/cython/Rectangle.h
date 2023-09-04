#ifndef RECTANGLE_H
#define RECTANGLE_H

#include <vector>

namespace shapes {
    class Rectangle {
        public:
            int x0, y0, x1, y1;
            std::vector<int> v;
            Rectangle();
            Rectangle(int x0, int y0, int x1, int y1);
            ~Rectangle();
            void setCoordinates(int x0, int y0, int x1, int y1);
            int getArea();
            void getSize(int* width, int* height);
            void move(int dx, int dy);
            std::vector<int> returnVector();
    };
}

#endif
