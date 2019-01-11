#include "header.h"

#define MAX_SOURCE_SIZE	16384
// wanted image size
#define DESIRED_WIDTH 400
#define DESIRED_HEIGHT 800

int main() {
    unsigned char *imageGray, *imageRGB;
    unsigned *energy;
    unsigned width, height, pitchGray, pitchRGB, imageSize, globalWidth, globalHeight;
    int *backtrack, i;
    double elapsedCPU;
    struct timespec start{}, finish{};

    // run tests
    /*if (test() != 0) {
        return 1;
    }*/

    // image reading
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, "../images/image.jpg", 0);
    FIBITMAP *imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
    width = FreeImage_GetWidth(imageBitmapGrey);
    height = FreeImage_GetHeight(imageBitmapGrey);
    pitchGray = FreeImage_GetPitch(imageBitmapGrey);
    pitchRGB = FreeImage_GetPitch(imageBitmap);
    imageSize = height * width;

    // save image size
    globalWidth = width;
    globalHeight = height;

    // memory allocation
    imageGray = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    imageRGB = (unsigned char *) malloc(3 * imageSize * sizeof(unsigned char));
    energy = (unsigned *) malloc(imageSize * sizeof(unsigned));
    backtrack = (int *) malloc(height * sizeof(int));

    // load image to memory (gray for sobel and rgb for carving)
    FreeImage_ConvertToRawBits(imageGray, imageBitmapGrey, pitchGray, 8,
            0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_ConvertToRawBits(imageRGB, imageBitmap, pitchRGB, 24,
            0xFF, 0xFF, 0xFF, TRUE);

    // remove read image
    FreeImage_Unload(imageBitmapGrey);
    FreeImage_Unload(imageBitmap);

    // start measuring time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // find and delete seams (width)
    for (i = 0; i < (globalWidth - DESIRED_WIDTH); i++) {
        sobelCPU(imageGray, energy, width, height);
        cumulativeCPU(energy, width, height);
        findSeam(energy, backtrack, width, height);
        deleteSeam(imageGray, imageRGB, backtrack, width, height);
        width--;
    }

    // rotate images
    FIBITMAP *rotatedGrayImage = FreeImage_ConvertFromRawBits(imageGray, width,
            height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
    FIBITMAP *rotatedRBGImage = FreeImage_ConvertFromRawBits(imageRGB, width,
            height, width*3, 24, 0xFF, 0xFF, 0xFF, TRUE);

    rotatedGrayImage = FreeImage_Rotate(rotatedGrayImage, 90, NULL);
    rotatedRBGImage = FreeImage_Rotate(rotatedRBGImage, 90, NULL);

    FreeImage_ConvertToRawBits(imageGray, rotatedGrayImage, height, 8,
            0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_ConvertToRawBits(imageRGB, rotatedRBGImage, height*3, 24,
            0xFF, 0xFF, 0xFF, TRUE);

    // find and delete seams (height)
    for (i = 0; i < (globalHeight - DESIRED_HEIGHT); i++) {
        sobelCPU(imageGray, energy, height, width);
        cumulativeCPU(energy, height, width);
        findSeam(energy, backtrack, height, width);
        deleteSeam(imageGray, imageRGB, backtrack, height, width);
        height--;
    }

    // stop
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsedCPU = (finish.tv_sec - start.tv_sec);
    elapsedCPU += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // rotate image back and save it
    FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageRGB, height,
            width, height*3, 24, 0xFF, 0xFF, 0xFF, TRUE);
    imageOutBitmap = FreeImage_Rotate(imageOutBitmap, -90, NULL);
    FreeImage_Save(FIF_PNG, imageOutBitmap, "../images/cpu_cut_image.png", 0);
    FreeImage_Unload(imageOutBitmap);

    // print results
    printf("Resized image from %dx%d to %dx%d.\n", globalWidth, globalHeight, width, height);
    printf("Calculation time: %f.", elapsedCPU);

    // free memory
    free(imageGray);
    free(imageRGB);
    free(energy);
    free(backtrack);

    return 0;
}

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