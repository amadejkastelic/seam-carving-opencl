//
// Created by amadej on 29. 12. 18.
// OpenCL kernel file
//
/**
 * Find minimum of three numbers.
 * @param x Number 1.
 * @param y Number 2.
 * @param z Number 3.
 * @return Minimum of input numbers.
 */
inline unsigned minimum(unsigned x, unsigned y, unsigned z) {
    return min(min(x, y), z);
}

inline unsigned getPixel(__global unsigned char *image, int width, int height, int y, int x, unsigned edge) {
    if (x < 0 || x >= width)
        return edge;
    if (y < 0 || y >= height)
        return edge;
    return image[y * width + x];
}

inline unsigned getPixelUnsigned(__global unsigned *image, int width, int height, int y, int x, unsigned edge) {
    if (x < 0 || x >= width)
        return edge;
    if (y < 0 || y >= height)
        return edge;
    return image[y * width + x];
}

inline unsigned getCachedPixel(__local unsigned char *image, int width, int height, int y, int x,
                               __global unsigned char *globalImage, int globalWidth, int globalHeight, int globalY,
                               int globalX) {
    // boundary case
    if (x < 0 || x >= width)
        return getPixel(globalImage, globalWidth, globalHeight, globalY, globalX, 0);
    if (y < 0 || y >= height)
        return getPixel(globalImage, globalWidth, globalHeight, globalY, globalX, 0);
    return image[y * width + x];
}

inline unsigned getCachedPixelUnsigned(__local unsigned *image, int y, int x, int cacheWidth, int cacheHeight) {
    return image[y * cacheWidth +  x];
}

/**
 * Find index of minimum element (util function for findSeam kernel).
 * @param image Input image (global).
 * @param width Image width.
 * @param height Image height.
 * @param y Row index.
 * @param x Column index.
 * @param len Number of how many elements to check.
 * @return Index of minimum element.
 */
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
 * @param imageOut Energy of image (result).
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

    index = row * width + j;
    cumulative[index] = cumulative[index] + minimum(
            getPixelUnsigned(cumulative, width, height, row + 1, j - 1, UINT_MAX),
            getPixelUnsigned(cumulative, width, height, row + 1, j, UINT_MAX),
            getPixelUnsigned(cumulative, width, height, row + 1, j + 1, UINT_MAX)
    );
}

__kernel void
cumulativeTrapezoid1(__global unsigned *cumulative, __local unsigned *cache, int width, int height, int numGroup) {
    int k;
    int i = get_global_id(0);
    int j = get_global_id(1);

    // calculate correct ids
    int globalI = i + numGroup * get_global_size(0);
    int globalJ = j + get_group_id(1) * get_local_size(1);

    int y = get_local_id(0);
    int x = get_local_id(1);

    int cacheHeight = get_local_size(0);
    int cacheWidth = get_local_size(1);
    int localCacheWidth = cacheWidth + 2*(cacheHeight/2)+2;

    cache[y*localCacheWidth + (x + cacheHeight/2+1)] = getPixelUnsigned(cumulative, width, height, globalI, globalJ, UINT_MAX);

    if (y == cacheHeight-1) {
        cache[(y+1)*localCacheWidth + (x + cacheHeight/2+1)] = getPixelUnsigned(cumulative, width, height, globalI+1, globalJ, UINT_MAX);

        if (x == 0) {
            cache[(y+1)*localCacheWidth + (x-1 + cacheHeight/2+1)] = getPixelUnsigned(cumulative, width, height, globalI+1, globalJ-1, UINT_MAX);
        } else if (x == cacheWidth-1) {
            cache[(y+1)*localCacheWidth + (x+1 + cacheHeight/2+1)] = getPixelUnsigned(cumulative, width, height, globalI+1, globalJ+1, UINT_MAX);
        }
    }

    if (globalJ == 0) {
        cache[y*localCacheWidth + cacheHeight/2+1 - 1] = UINT_MAX;
    }

    // mirror thread indexes
    if (get_group_id(1) != 0 && y - x > cacheHeight / 2.0f) {
        globalI = cacheHeight * numGroup + cacheHeight - 1 - y;
        globalJ = cacheWidth * (get_group_id(1) * 2 - 1) + cacheWidth - 1 - x;
        y = cacheHeight - 1 - y;
        x = -x - 1 + cacheHeight/2+1;

        cache[y*localCacheWidth + x] = getPixelUnsigned(cumulative, width, height, globalI, globalJ, UINT_MAX);

        for (k = 1; k <= 2 && y == x-1; k++) {
            cache[(y+k)*localCacheWidth + x] = getPixelUnsigned(cumulative, width, height, globalI+k, globalJ, UINT_MAX);
        }

        if (y == 0 && x == 1) {
            cache[(y+1)*localCacheWidth + x - 1] = getPixelUnsigned(cumulative, width, height, globalI + 1, globalJ - 1, UINT_MAX);
        }
    } else if (get_group_id(1) != get_global_size(1) / cacheWidth && x + y >= cacheWidth + cacheHeight / 2.0f - 1) {
        globalI = cacheHeight * numGroup + cacheHeight - 1 - y;
        globalJ = cacheWidth * (get_group_id(1) * 2 + 1) + cacheWidth - 1 - x;
        y = cacheHeight - 1 - y;
        x = cacheWidth - 1 + (cacheWidth - 1 - x)+ cacheHeight/2+2;

        cache[y*localCacheWidth + x] = getPixelUnsigned(cumulative, width, height, globalI, globalJ, UINT_MAX);

        for (k = 1; k <= 2 && y == localCacheWidth-2-x; k++) {
            cache[(y+k)*localCacheWidth + x] = getPixelUnsigned(cumulative, width, height, globalI+k, globalJ, UINT_MAX);
        }

        if (y == 0 && x == localCacheWidth-2) {
            cache[(y+1)*localCacheWidth + x + 1] = getPixelUnsigned(cumulative, width, height, globalI + 1, globalJ + 1, UINT_MAX);
        }
    } else {
        x = x + cacheHeight/2+1;
    }

    int localIndex = y * localCacheWidth + x;

    int globalIndex = globalI * width + globalJ;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (k = cacheHeight - 1; k >= 0; k--) {
        if (y == k && globalI < height-1 && globalJ < width) {
            cache[localIndex] += minimum(
                    getCachedPixelUnsigned(cache, y + 1, x - 1, localCacheWidth, cacheHeight),
                    getCachedPixelUnsigned(cache, y +1, x, localCacheWidth, cacheHeight),
                    getCachedPixelUnsigned(cache, y + 1, x + 1, localCacheWidth, cacheHeight)
            );
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if (globalI < height-1 && globalJ < width) {
        cumulative[globalIndex] = cache[localIndex];
    }
}

__kernel void
cumulativeTrapezoid2(__global unsigned *cumulative, __local unsigned *cache, int width, int height, int numGroup) {
    int k;
    int i = get_global_id(0); // 0-13
    int j = get_global_id(1); // 0 : width/2

    // calculate correct ids
    int globalI = i + numGroup * get_global_size(0); //0-- height
    int globalJ = j + (get_group_id(1) + 1) * get_local_size(1); // 0 - width

    int y = get_local_id(0);  //0:15
    int x = get_local_id(1); // 0:16

    int cacheHeight = get_local_size(0);
    int cacheWidth = get_local_size(1);
    int localCacheWidth = cacheWidth + 2*(cacheHeight/2)+2;

    if (globalJ == width-1) {
        cache[y*localCacheWidth + x + cacheHeight/2+1 + 1] = UINT_MAX;

        if (y == cacheHeight-1) {
            cache[(y+1)*localCacheWidth + x + cacheHeight/2+1 + 1] = UINT_MAX;
        }
    }

    // mirror thread indexes
    if (y + x <= cacheHeight / 2.0f - 1) {
        globalI = cacheHeight * numGroup + cacheHeight - 1 - y;
        globalJ = cacheWidth * ((get_group_id(1) * 2 + 1) - 1) + cacheWidth - 1 - x;
        y = cacheHeight - 1 - y;
        x = -x - 1 + cacheHeight/2+1;
    } else if (get_group_id(1) + 1 != get_global_size(1) / cacheWidth && y + cacheWidth - 1 - x <= cacheHeight / 2.0f -1) {
        globalI = cacheHeight * numGroup + cacheHeight - 1 - y;
        globalJ = cacheWidth * ((get_group_id(1) * 2 + 1) + 1) + cacheWidth - 1 - x;
        y = cacheHeight - 1 - y;
        x = cacheWidth - 1 + (cacheWidth - 1 - x)+ cacheHeight/2+2;
    } else {
        x = x + cacheHeight/2+1;
    }


    int localIndex = y * localCacheWidth + x;

    int globalIndex = globalI * width + globalJ;

    // copy data to local memory
    cache[localIndex] = getPixelUnsigned(cumulative, width, height, globalI, globalJ, UINT_MAX);

    if (y == cacheHeight-1 && x == 1 && globalI < height-1) {
        for (k = 0; k < localCacheWidth; k++) {
            cache[(y+1)*localCacheWidth + k] = getPixelUnsigned(cumulative, width, height, globalI+1, globalJ-1+k, UINT_MAX);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (k = cacheHeight - 1; k >= 0; k--) {
        if (y == k && globalI < height-1 && globalJ < width) {
            cache[localIndex] += minimum(
                    getCachedPixelUnsigned(cache, y + 1, x - 1, localCacheWidth, cacheHeight),
                    getCachedPixelUnsigned(cache, y +1, x, localCacheWidth, cacheHeight),
                    getCachedPixelUnsigned(cache, y + 1, x + 1, localCacheWidth, cacheHeight)
            );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (globalI < height-1 && globalJ < width) {
        cumulative[globalIndex] = cache[localIndex];
    }
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

/**
 * Calculates last step of reduction and finds the seam to delete.
 * @param cumulative Array of cumulatives.
 * @param reduction Array of findMin's result values.
 * @param reductionIndex Array of findMin's result indexes.
 * @param cache Local array for reduction values cache.
 * @param cacheIndex Local array for reduction indexes cache.
 * @param reductionWidth Size of reduction and reductionIndex array.
 * @param width Image width.
 * @param height Image height.
 * @param backtrack Array of indexes for storing the result (seam to remove).
 */
__kernel void findSeam(__global unsigned *cumulative, __global unsigned *reduction, __global unsigned *reductionIndex,
                       __local unsigned *cache, __local int *cacheIndex, unsigned reductionWidth, int width, int height,
                       __global int *backtrack) {
    int j = get_global_id(0);

    // copy global memory to local
    if (j < reductionWidth) {
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
            backtrack[i] = indexOfMin(cumulative, width, height, i, backtrack[i - 1] - 1, 3);
        }
    }
}

/**
 * Deletes found seam.
 * @param gray Array with gray image.
 * @param grayCopy Array for cut 8-bit image.
 * @param RGB Array with 24-bit image.
 * @param RGBCopy Array for cut 24-bit image.
 * @param backtrack Seam to delete.
 * @param width Image width.
 * @param height Image height.
 */
__kernel void deleteSeam(__global unsigned char *gray, __global unsigned char *grayCopy,
                         __global unsigned char *RGB, __global unsigned char *RGBCopy, __global int *backtrack,
                         int width, int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= height || j >= width) {
        return;
    }

    int chunk;
    int start = backtrack[i];
    if (j < start) {
        chunk = i;
    } else if (j == start) {
        return;
    } else {
        chunk = i + 1;
    }

    int index = i * width + j;
    grayCopy[index - chunk] = gray[index];

    if (j < width) {
        index = i * (width * 3) + (j * 3);
        for (int k = index; k < index + 3; k++) {
            RGBCopy[k - (chunk * 3)] = RGB[k];
        }
    }
}

__kernel void
transpose(__global unsigned char *srcGray, __global unsigned char *destGray, __local unsigned char *cacheGray,
          __global unsigned char *srcRGB, __global unsigned char *destRGB, __local unsigned char *cacheRGB, int width,
          int height) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int y = get_local_id(0);
    int x = get_local_id(1);

    int localSize = get_local_size(0);

    // copy to local
    if (i < height && j < width) {
        cacheGray[y * localSize + x] = srcGray[i * width + j];

        int cacheIndex = y * 3 * localSize + 3 * x;
        int globalIndex = i * width * 3 + j * 3;
        for (int k = 0; k < 3; k++) {
            cacheRGB[cacheIndex + k] = srcRGB[globalIndex + k];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // write to global
    i = get_group_id(1) * localSize + y;
    j = get_group_id(0) * localSize + x;
    if (i < width && j < height) {
        destGray[i * height + j] = cacheGray[x * localSize + y];

        int cacheIndex = x * 3 * localSize + 3 * y;
        int globalIndex = i * height * 3 + 3 * j;
        for (int k = 0; k < 3; k++) {
            destRGB[globalIndex + k] = cacheRGB[cacheIndex + k];
        }
    }
}