//
// Created by amadej on 11. 01. 19.
//

#include "header.h"

void sobelCPU(unsigned char *imageIn, unsigned *imageOut, int width, int height) {
    int i, j;
    int Gx, Gy;
    unsigned tempPixel;

    for (i = 0; i < (height); i++) {
        for (j = 0; j < (width); j++) {
            Gx = -getPixelImage(imageIn, width, height, i - 1, j - 1, 0) -
                 2 *getPixelImage(imageIn, width, height, i - 1, j, 0) -
                 getPixelImage(imageIn, width, height, i - 1, j + 1, 0) +
                 getPixelImage(imageIn, width, height, i + 1, j - 1, 0) +
                 2 * getPixelImage(imageIn, width, height, i + 1, j, 0) +
                 getPixelImage(imageIn, width, height, i + 1, j + 1, 0);

            Gy = -getPixelImage(imageIn, width, height, i - 1, j - 1, 0) -
                 2 * getPixelImage(imageIn, width, height, i, j - 1, 0) -
                 getPixelImage(imageIn, width, height, i + 1, j - 1, 0) +
                 getPixelImage(imageIn, width, height, i - 1, j + 1, 0) +
                 2 * getPixelImage(imageIn, width, height, i, j + 1, 0) +
                 getPixelImage(imageIn, width, height, i + 1, j + 1, 0);

            // normalization
            //Gx /= 8;
            //Gy /= 8;

            tempPixel = (unsigned) sqrt((float) (Gx * Gx + Gy * Gy));

            if (tempPixel > 255) {
                imageOut[i * width + j] = 255;
            } else {
                imageOut[i * width + j] = tempPixel;
            }
        }
    }
}

void cumulativeCPU(unsigned *image, int width, int height) {
    int i, j, index;

    for (i = height-2; i >= 0; i--) {
        for (j = 0; j < width; j++) {
            index = i*width + j;
            image[index] = image[index] + min(
                    getPixel(image, width, height, i+1, j-1, UINT_MAX),
                    getPixel(image, width, height, i+1, j, UINT_MAX),
                    getPixel(image, width, height, i+1, j+1, UINT_MAX)
            );
        }
    }
}

void findSeam(unsigned *image, int *backtrack, int width, int height) {
    int i, temp;

    // find min in top row
    temp = image[0];
    backtrack[0] = 0;
    for (i = 1; i < width; i++) {
        if (image[i] < temp) {
            backtrack[0] = i;
            temp = image[i];
        }
    }

    // find seam
    for (i = 1; i < height; i++) {
        backtrack[i] =
                indexOfMin(image, width, height, i, backtrack[i-1]-1, 3);
    }
}

void deleteSeam(unsigned char *gray, unsigned char *RGB, const int *backtrack, int width, int height) {
    int i, j, k, index, chunk;

    chunk = 0; // number of deleted pixels
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (j == backtrack[i]) {
                chunk++;
                continue;
            }

            // if no pixels deleted yet
            if (chunk == 0) {
                continue;
            }

            // move pixel back for chunk size (gray image)
            index = i*width + j; // index of current pixel
            gray[index - chunk] = gray[index];

            // move pixel back for chunk size (RGB image)
            index = i * (width*3) + (j*3); // index of R in current pixel
            for (k = index; k < index + 3; k++) {
                RGB[k - (chunk*3)] = RGB[k];
            }
        }
    }
}
