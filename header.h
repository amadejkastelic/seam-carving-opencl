//
// Created by amadej on 23. 12. 18.
// Headers
//

#ifndef SEMINAR_HEADER_H
#define SEMINAR_HEADER_H

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include "FreeImage.h"

#define _CRT_SECURE_NO_WARNINGS

int min(unsigned x, unsigned y, unsigned z);

int max(unsigned x, unsigned y, unsigned z);

unsigned getPixel(unsigned *image, int width, int height, int y, int x, unsigned edge);

unsigned getPixelImage(unsigned char *image, int width, int height, int y, int x, unsigned edge);

int indexOfMin(unsigned *image, int width, int height, int y, int x, int len);

void sobelCPU(unsigned char *imageIn, unsigned *imageOut, int width, int height);

void cumulativeCPU(unsigned *image, int width, int height);

void findSeam(unsigned *image, int *backtrack, int width, int height);

void deleteSeam(unsigned char *gray, unsigned char *RGB, const int *backtrack, int width, int height);

void resizeImageSerial(const char *imagePath);

void resizeImageParallel(const char *imagePath);

void saveUnsignedImage(const unsigned int *image, int width, int height, const char *path);

void saveImage(unsigned char *image, int width, int height, unsigned depth, const char *path);

int nearestPower(int num);

unsigned nearestMultipleOf(unsigned num, int multiple);

double getTime(timespec *start, timespec *finish);

#endif //SEMINAR_HEADER_H
