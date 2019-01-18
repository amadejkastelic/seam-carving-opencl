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

unsigned getPixel(unsigned *image, int width, int height, int y, int x, unsigned edge) {
    if (x < 0 || x >= width) {
        return edge;
    }

    if (y < 0 || y >= height) {
        return edge;
    }

    return image[y*width + x];
}

unsigned getPixelImage(unsigned char *image, int width, int height, int y, int x, unsigned edge) {
    if (x < 0 || x >= width) {
        return edge;
    }

    if (y < 0 || y >= height) {
        return edge;
    }

    return image[y*width + x];
}

int indexOfMin(unsigned *image, int width, int height, int y, int x, int len) {
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

void saveUnsignedImage(const unsigned int *image, int width, int height, const char *path) {
    int imageSize = width*height;

    auto *temp = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    for (int i = 0; i < imageSize; i++) {
        temp[i] = (unsigned char) image[i];
    }

    FIBITMAP *imageOut = FreeImage_ConvertFromRawBits(temp, width,
            height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, imageOut, path, 0);
    FreeImage_Unload(imageOut);

    free(temp);
}

void saveImage(unsigned char *image, int width, int height, unsigned depth, const char *path) {
    FIBITMAP *imageOut = FreeImage_ConvertFromRawBits(image, width,
            height, width*(depth/8), depth, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, imageOut, path, 0);
    FreeImage_Unload(imageOut);
}

// source: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
int nearestPower(int num) {
    num--;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    num++;

    return num;
}

unsigned nearestMultipleOf(unsigned num, int multiple) {
    return ((num + multiple - 1) / multiple) * multiple;
}
