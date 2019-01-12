//
// Created by amadej on 29. 12. 18.
// OpenCL kernel file
//
inline unsigned minimum(unsigned x, unsigned y, unsigned z) {
    return min(min(x, y), z);
}

inline unsigned getPixel(__global unsigned char *image, int width, int height, int y, int x, unsigned edge) {
    if (x < 0 || x >= width)
        return edge;
    if (y < 0 || y >= height)
        return edge;
    return image[y*width + x];
}

inline unsigned getPixelUnsigned(__global unsigned *image, int width, int height, int y, int x, unsigned edge) {
    if (x < 0 || x >= width)
        return edge;
    if (y < 0 || y >= height)
        return edge;
    return image[y*width + x];
}

inline unsigned getCachedPixel(__local unsigned char *image, int width, int height, int y, int x,
                          __global unsigned char *globalImage, int globalWidth, int globalHeight, int globalY, int globalX) {
    // boundary case
    if (x < 0 || x >= width)
        return getPixel(globalImage, globalWidth, globalHeight, globalY, globalX, 0);
    if (y < 0 || y >= height)
        return getPixel(globalImage, globalWidth, globalHeight, globalY, globalX, 0);
    return image[y*width + x];
}

inline int indexOfMin(__global unsigned *image, int width, int height, int y, int x, int len) {
    int i, index;
    unsigned min, temp;

    index = x;
    min = getPixelUnsigned(image, width, height, y, index, UINT_MAX);
    for (i = x + 1; i < x + len; i++) {
        temp = getPixelUnsigned(image, width, height, y, i, UINT_MAX);
        if (temp < min) {
            min = temp;
            index = i;
        }
    }

    return index;
}

/**
 * Calculate energy of specified image.
 * @param imageIn Input image.
 * @param imageOut Energy of image.
 * @param cached Local memory for calculation.
 * @param width Image width.
 * @param height Image height.
 */
__kernel void sobel(__global unsigned char *imageIn, __global unsigned *imageOut,
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

    cached[index] = getPixel(imageIn, width, height, i, j, 0);

    if (y == cacheHeight) {
        cached[(y+1) * cacheWidth + x] = getPixel(imageIn, width, height, y+1, x, 0);
    }
    if (x == cacheWidth) {
        cached[index + 1] = getPixel(imageIn, width, height, y, x+1, 0);
    }
    if (y == cacheHeight && x == cacheWidth) {
        cached[(y+1) * cacheWidth + x + 1] = getPixel(imageIn, width, height, y+1, x+1, 0);
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

/**
 * Calculate cumulative of energy.
 * @param cumulative Array with energy.
 * @param width Image width.
 * @param height Image height.
 * @param row Row to calculate.
 */
__kernel void cumulativeBasic(__global unsigned *cumulative, int width, int height, int row) {
    int index, j = get_global_id(0);

    if (j >= width) {
        return;
    }

    index = row*width + j;
    cumulative[index] = cumulative[index] + minimum(
            getPixelUnsigned(cumulative, width, height, row+1, j-1, UINT_MAX),
            getPixelUnsigned(cumulative, width, height, row+1, j, UINT_MAX),
            getPixelUnsigned(cumulative, width, height, row+1, j+1, UINT_MAX)
    );
}

/**
 * Find minimum using reduction.
 * @param cumulative Array of calculated cumulatives.
 * @param result Array with reduction result.
 * @param resultIndex Array with result's indexes.
 * @param cache Local array for each block.
 * @param cacheIndex Array with cache's indexes.
 * @param width Image width.
 */
__kernel void findMin(__global unsigned *cumulative, __global unsigned *result, __global unsigned *resultIndex,
        __local unsigned *cache, __local int *cacheIndex, int width) {
    int j = get_global_id(0);
    int y = get_local_id(0);
    int gid = get_group_id(0);

    // copy global memory to local
    if (j < width) {
        cache[y] = cumulative[j];
        cacheIndex[y] = j;
    } else {
        cache[y] = UINT_MAX;
        cacheIndex[y] = -1;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int offset = get_local_size(0) / 2;
    while (offset > 0) {
        if (y < offset) {
            if (cache[y + offset] < cache[y]) {
                cache[y] = cache[y + offset];
                cacheIndex[y] = cacheIndex[y + offset];
            }
        }

        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // copy local results to global memory
    if (y == 0) {
        result[gid] = cache[0];
        resultIndex[gid] = cacheIndex[0];
    }
}

__kernel void findSeam(__global unsigned *cumulative, __global unsigned *reduction, __global unsigned *reductionIndex,
        __local unsigned *cache, __local int *cacheIndex, unsigned reductionWidth, int width, int height,
        __global int *backtrack) {
    int j = get_global_id(0);

    // copy global memory to local
    if (j  < reductionWidth) {
        cache[j] = reduction[j];
        cacheIndex[j] = reductionIndex[j];
    } else {
        cache[j] = UINT_MAX;
        cacheIndex[j] = -1;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    int offset = get_global_size(0) / 2;
    while (offset > 0) {
        if (j < offset) {
            if (cache[j + offset] < cache[j]) {
                cache[j] = cache[j + offset];
                cacheIndex[j] = cacheIndex[j + offset];
            }
        }

        offset /= 2;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // serial algorithm
    if (j == 0) {
        backtrack[0] = cacheIndex[0];
        for (int i = 1; i < height; i++) {
            backtrack[i] = indexOfMin(cumulative, width, height, i, backtrack[i-1]-1, 3);
        }
    }
}

__kernel void deleteSeam(__global unsigned char *gray, __global unsigned char *grayCopy,
        __global unsigned char *RGB, __global unsigned char *RGBCopy, __global int *backtrack,
        int width, int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= height || j >= width) {
        return;
    }

    int chunk;
    int pixel = backtrack[i];
    if (j < pixel) {
        chunk = i;
    } else if (j == pixel) {
        return;
    } else {
        chunk = i+1;
    }

    int index = i*width + j;
    grayCopy[index - chunk] = gray[index];

    index = i * (width*3) + (j*3);
    for (int k = index; k < index + 3; k++) {
        RGBCopy[k - (chunk*3)] = RGB[k];
    }
}