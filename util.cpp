//
// Created by amadej on 29. 12. 18.
// Utility functions
//

#include "header.h"

int min(unsigned x, unsigned y, unsigned z) {
    return (x = (x <= y) ? x : y) <= z ? x : z;
}

int max(unsigned x, unsigned y, unsigned z) {
    return (x = (x >= y) ? x : y) >= z ? x : z;
}

unsigned getPixel(unsigned *image, int width, int height, int y, int x,
                  unsigned edge) {
    if (x < 0 || x >= width) {
        return edge;
    }

    if (y < 0 || y >= height) {
        return edge;
    }

    return image[y*width + x];
}

unsigned getPixelImage(unsigned char *image, int width, int height,
                       int y, int x, unsigned edge) {
    if (x < 0 || x >= width) {
        return edge;
    }

    if (y < 0 || y >= height) {
        return edge;
    }

    return image[y*width + x];
}

int indexOfMin(unsigned *image, int width, int height,
               int y, int x, int len) {
    int i, index;
    unsigned min, temp;

    index = x;
    min = getPixel(image, width, height, y, index, UINT_MAX);
    for (i = x + 1; i < x + len; i++) {
        temp = getPixel(image, width, height, y, i, UINT_MAX);
        if (temp < min) {
            min = temp;
            index = i;
        }
    }

    return index;
}
