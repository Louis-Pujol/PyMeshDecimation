#include <iostream>
#include <vector>
#include "Rectangle.h"

namespace shapes {

    // Default constructor
    Rectangle::Rectangle () {}

    void Rectangle::setCoordinates(int x0, int y0, int x1, int y1){
        this->x0 = x0;
        this->y0 = y0;
        this->x1 = x1;
        this->y1 = y1;

        this->v = std::vector<int>();
        this->v.push_back(x0);
        this->v.push_back(y0);
        this->v.push_back(x0);
        this->v.push_back(x1);
    }

    // Overloaded constructor
    Rectangle::Rectangle (int x0, int y0, int x1, int y1) {
        this->x0 = x0;
        this->y0 = y0;
        this->x1 = x1;
        this->y1 = y1;

    }

    // Destructor
    Rectangle::~Rectangle () {}

    // Return the area of the rectangle
    int Rectangle::getArea () {
        // return (this->x1 - this->x0) * (this->y1 - this->y0);
        // this->v = std::vector<int>();
        // this->v.push_back(x0);
        // this->v.push_back(x1);
        // this->v.push_back(y0);
        // this->v.push_back(y1);

        return (this->v.at(1) - this->v.at(0)) * (this->v.at(3) - this->v.at(2));
    }

    // Get the size of the rectangle.
    // Put the size in the pointer args
    void Rectangle::getSize (int *width, int *height) {
        (*width) = x1 - x0;
        (*height) = y1 - y0;
    }

    // Move the rectangle by dx dy
    void Rectangle::move (int dx, int dy) {
        this->x0 += dx;
        this->y0 += dy;
        this->x1 += dx;
        this->y1 += dy;
    }

    // Return a vector of the rectangle's coordinates
    std::vector<int> Rectangle::returnVector () {
        std::vector<int> v;
        v.push_back(this->x0);
        v.push_back(this->y0);
        v.push_back(this->x1);
        v.push_back(this->y1);
        return v;
    }
}