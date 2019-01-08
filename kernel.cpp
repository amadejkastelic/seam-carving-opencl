inline int getPixel(__global unsigned char *image, int width, int height, int y, int x) {
    if (x < 0 || x >= width)
        return 0;
    if (y < 0 || y >= height)
        return 0;
    return image[y*width + x];
}

inline int getCachedPixel(__local unsigned char *image, int width, int height, int y, int x,
                          __global unsigned char *globalImage, int globalWidth, int globalHeight, int globalY, int globalX) {
    // boundary case
    if (x < 0 || x >= width)
        return getPixel(globalImage, globalWidth, globalHeight, globalY, globalX);
    if (y < 0 || y >= height)
        return getPixel(globalImage, globalWidth, globalHeight, globalY, globalX);
    return image[y*width + x];
}

__kernel void sobel(__global unsigned char *imageIn, __global unsigned char *imageOut,
                    __local unsigned char *cached, int width, int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= height || j >= width) {
        return;
    }

    // save all to local memory
    int y = get_local_id(0);
    int x = get_local_id(1);

    int cacheHeight = get_local_size(0);
    int cacheWidth = get_local_size(1);

    int index = y * cacheWidth + x;

    cached[index] = getPixel(imageIn, width, height, i, j);

    if (y == cacheHeight) {
        cached[(y+1) * cacheWidth + x] = getPixel(imageIn, width, height, y+1, x);
    }
    if (x == cacheWidth) {
        cached[index + 1] = getPixel(imageIn, width, height, y, x+1);
    }
    if (y == cacheHeight && x == cacheWidth) {
        cached[(y+1) * cacheWidth + x + 1] = getPixel(imageIn, width, height, y+1, x+1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int Gx, Gy;
    int tempPixel;

    Gx = -getCachedPixel(cached, cacheWidth, cacheHeight, y - 1, x - 1, imageIn, width, height, i - 1, j - 1) -
         2 * getCachedPixel(cached, cacheWidth, cacheHeight, y - 1, x, imageIn, width, height, i - 1, j) -
         getCachedPixel(cached, cacheWidth, cacheHeight, y - 1, x + 1, imageIn, width, height, i - 1, j + 1) +
         getCachedPixel(cached, cacheWidth, cacheHeight, y + 1, x - 1, imageIn, width, height, i + 1, j - 1) +
         2 * getCachedPixel(cached, cacheWidth, cacheHeight, y + 1, x, imageIn, width, height, i + 1, j) +
         getCachedPixel(cached, cacheWidth, cacheHeight, y + 1, x + 1, imageIn, width, height, i + 1, j + 1);

    Gy = -getCachedPixel(cached, cacheWidth, cacheHeight, y - 1, x - 1, imageIn, width, height, i - 1, j - 1) -
         2 * getCachedPixel(cached, cacheWidth, cacheHeight, y, x - 1, imageIn, width, height, i, j - 1) -
         getCachedPixel(cached, cacheWidth, cacheHeight, y + 1, x - 1, imageIn, width, height, i + 1, j - 1) +
         getCachedPixel(cached, cacheWidth, cacheHeight, y - 1, x + 1, imageIn, width, height, i - 1, j + 1) +
         2 * getCachedPixel(cached, cacheWidth, cacheHeight, y, x + 1, imageIn, width, height, i, j + 1) +
         getCachedPixel(cached, cacheWidth, cacheHeight, y + 1, x + 1, imageIn, width, height, i + 1, j + 1);

    tempPixel = sqrt((float) (Gx * Gx + Gy * Gy));
    if (tempPixel > 255) {
        imageOut[i * width + j] = 255;
    } else {
        imageOut[i * width + j] = tempPixel;
    }
}
