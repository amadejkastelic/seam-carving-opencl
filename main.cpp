#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>

#include "FreeImage.h"
#include "header.h"
#include "tests.h"

#define MAX_SOURCE_SIZE	16384

// wanted image size
#define WANTED_WIDTH 800;
#define WANTED_HEIGHT 600;

int main() {
    unsigned char *imageIn, *image;
    unsigned *imageOut;
    unsigned width, height, pitch, imageSize;
    int i, *backtrack;

    // run tests
    /*if (test() != 0) {
        return 1;
    }*/

    // image reading
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_PNG, "../image.png", 0);
    FIBITMAP *imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
    width = FreeImage_GetWidth(imageBitmapGrey);
    height = FreeImage_GetHeight(imageBitmapGrey);
    pitch = ((32 * width + 31) / 32) * 4;
    imageSize = height * width;

    // memory allocation
    image = (unsigned char *) malloc(3 * imageSize * sizeof(unsigned char));
    imageIn = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    imageOut = (unsigned *) malloc(imageSize * sizeof(unsigned));
    backtrack = (int *) malloc(height * sizeof(int));

    // load image to memory
    FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, width, 8,
            0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_ConvertToRawBits(image, imageBitmap, pitch, 32,
            0xFF, 0xFF, 0xFF, TRUE);

    // remove read image
    FreeImage_Unload(imageBitmapGrey);
    FreeImage_Unload(imageBitmap);

    // find and delete seams
    for (i = 0; i < 400; i++) {
        sobelCPU(imageIn, imageOut, width, height);
        cumulativeCPU(imageOut, width, height);
        findSeam(imageOut, image, width, height);
        //deleteSeamImage(imageIn, backtrack, width, height);
        deleteSeamColorImage(image, imageIn, width, height);
        width--;
    }

    // save new image
    FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageIn, width,
            height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, imageOutBitmap, "../cpu_cut_image.png", 0);
    FreeImage_Unload(imageOutBitmap);

    // free memory
    free(image);
    free(imageIn);
    free(imageOut);

    return 0;
}

void sobelCPU(unsigned char *imageIn, unsigned *imageOut,
        int width, int height) {
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
            /*if (temp >= 255) {
                image[index] = 255;
            } else {
                image[index] = (unsigned char) temp;
            }*/
        }
    }
}

void findSeam(unsigned *image, unsigned char *imageOut,
        int width, int height) {
    int i, temp, index;

    // find min in top row
    temp = image[0];
    index = 0;
    for (i = 1; i < width; i++) {
        if (image[i] < temp) {
            index = i;
            temp = image[i];
        }
    }
    // set alpha bit of min pixel to 0
    imageOut[4*index - 1] = 0;

    // find seam
    for (i = 1; i < height; i++) {
        index = indexOfMin(image, width, height, i, index-1, 3);
        // set alpha bit of each pixel in seam to 0
        imageOut[i * (width*4) + (index*4) - 1] = 0;
    }
}

void deleteSeam(unsigned *image, int *backtrack, int width, int height) {
    int i, j, imageSize, index;

    imageSize = width * height;
    for (i = 0; i < height; i++) {
        index = i * width;
        for (j = index + backtrack[i]-i; j < imageSize-i-1; j++) {
            image[j] = image[j + 1];
        }
    }
}

void deleteSeamImage(unsigned char *image, int *backtrack, int width,
        int height) {
    int i, j, imageSize, index;

    imageSize = width * height;
    for (i = 0; i < height; i++) {
        index = i * width;
        for (j = index + backtrack[i]-i; j < imageSize-i-1; j++) {
            image[j] = image[j + 1];
        }
    }
}

void deleteSeamColorImage(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int i, j, k, index, chunk;

    chunk = 0;
    for (i = 0; i < height; i++) {
        for (j = 1; j < width+1; j++) {
            index = i * (width*4) + (j*4) - 1;
            if (rgb[index] == 0) {
                chunk+=4;
                continue;
            }

            if (chunk == 0) {
                continue;
            }

            // delete from rgb image
            index = index - 4 + 1;

            for (k = index; k < index + 4; k++) {
                rgb[index - chunk] = rgb[index];
            }

            // delete from gray image
            gray[index/4 - chunk/4] = gray[index/4];
        }
    }
}

void colorSeam(unsigned char *image, int *backtrack, int width, int height) {
    int i;

    for (i = 0; i < height; i++) {
        image[i*width + backtrack[i]] = 0;
    }
}