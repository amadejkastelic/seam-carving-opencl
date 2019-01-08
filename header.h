//
// Created by amadej on 23. 12. 18.
//

#ifndef SEMINAR_HEADER_H
#define SEMINAR_HEADER_H

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

void sobelCPU(unsigned char *imageIn, unsigned *imageOut,
              int width, int height);

void cumulativeCPU(unsigned *image, int width, int height);

void findSeam(unsigned *image, int *backtrack,
              int width, int height);

void deleteSeam(unsigned *image, int *backtrack,
                int width, int height);

void deleteSeamImage(unsigned char *image, int *backtrack,
                int width, int height);

void colorSeam(unsigned char *image, int *backtrack, int width, int height);

#endif //SEMINAR_HEADER_H
