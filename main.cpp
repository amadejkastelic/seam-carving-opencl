//
// Created by amadej on 29. 12. 18.
//
#include "header.h"

#define MAX_SOURCE_SIZE    16384*2
#define WORKGROUP_SIZE 16
// path to image
#define IMAGE_PATH "images/image.jpg"
// wanted image size
#define DESIRED_WIDTH 400
#define DESIRED_HEIGHT 800
// debug mode
#define DEBUG 0

int main() {
    // Parallel algorithm
    resizeImageParallel(IMAGE_PATH);

    // Serial algorithm
    //resizeImageSerial(IMAGE_PATH);

    return 0;
}

void resizeImageParallel(const char *imagePath) {
    unsigned char *imageGray, *imageRGB;
    unsigned width, height, pitchGray, pitchRGB, imageSize, globalWidth, globalHeight;
    int i, row;
    double elapsed;
    struct timespec start{}, finish{};
    cl_int ret;

    // read image
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, imagePath, 0);
    FIBITMAP *imageBitmapGray = FreeImage_ConvertToGreyscale(imageBitmap);
    width = FreeImage_GetWidth(imageBitmapGray);
    height = FreeImage_GetHeight(imageBitmapGray);
    pitchGray = FreeImage_GetPitch(imageBitmapGray);
    pitchRGB = FreeImage_GetPitch(imageBitmap);
    imageSize = height * width;

    globalWidth = width;
    globalHeight = height;

    // read and compile kernel
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kernel.cpp", "r");
    if (!fp) {
        fprintf(stderr, "Kernel file missing.\n");
        exit(1);
    }
    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    // memory allocation
    imageGray = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    imageRGB = (unsigned char *) malloc(3 * imageSize * sizeof(unsigned char));

    // opencl configuration
    cl_platform_id platform_id[10];
    cl_uint ret_num_platforms;
    char buf[100];
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    cl_device_id device_id[10];
    cl_uint ret_num_devices;
    ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
                         device_id, &ret_num_devices);

    clGetPlatformInfo(platform_id[0], CL_PLATFORM_NAME, sizeof(buf), &buf, NULL);

    printf("Devices: %d\n", ret_num_devices);
    printf("Platforms: %d\n", ret_num_platforms);
    printf("Platform name: %s\n", buf);

    clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, sizeof(buf), &buf, NULL);
    printf("Device name: %s\n", buf);

    clGetDeviceInfo(device_id[0], CL_DEVICE_OPENCL_C_VERSION, sizeof(buf), &buf, NULL);
    printf("OpenCL version: %s\n", buf);

    // get max workgroup size
    cl_uint max_work_item_dimensions;
    clGetDeviceInfo(device_id[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions),
                    &max_work_item_dimensions, NULL);
    auto *size = (size_t *) (malloc(sizeof(size_t) * max_work_item_dimensions));
    ret = clGetDeviceInfo(device_id[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions,
                          size, NULL);
    printf("Max work size: %lu\n", size[0]);

    cl_context context = clCreateContext(NULL, ret_num_devices, &device_id[0], NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // load image
    FreeImage_ConvertToRawBits(imageGray, imageBitmapGray, pitchGray, 8, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_ConvertToRawBits(imageRGB, imageBitmap, pitchRGB, 24, 0xFF, 0xFF, 0xFF, TRUE);

    // prepare program
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source_str, NULL, &ret);

    // compile program
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

    // logs
    size_t build_log_len;
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &build_log_len);
    char *build_log = (char *) malloc(sizeof(char) * (build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);

    // prepare variables for group sizes
    // 1D
    size_t localSize, numGroups, globalSize;
    // 2D
    size_t local_size[2], num_groups[2], global_size[2];

    // prepare kernels
    cl_kernel sobelKernel = clCreateKernel(program, "sobel", &ret);
    cl_kernel cumulativeBasicKernel = clCreateKernel(program, "cumulativeBasic", &ret);
    cl_kernel findMinKernel = clCreateKernel(program, "findMin", &ret);
    cl_kernel findSeamKernel = clCreateKernel(program, "findSeam", &ret);
    cl_kernel deleteSeamKernel = clCreateKernel(program, "deleteSeam", &ret);
    cl_kernel transposeKernel = clCreateKernel(program, "transpose", &ret);
    cl_kernel trapezoid1CumulativeKernel = clCreateKernel(program, "cumulativeTrapezoid1", &ret);
    cl_kernel trapezoid2CumulativeKernel = clCreateKernel(program, "cumulativeTrapezoid2", &ret);

    // allocate gpu memory (we have enough - 8GB)
    cl_mem gray_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         imageSize * sizeof(unsigned char), NULL, &ret);
    cl_mem energy_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                           imageSize * sizeof(unsigned), NULL, &ret);
    cl_mem gray_copy_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              imageSize * sizeof(unsigned char), NULL, &ret);
    cl_mem RGB_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                        imageSize * 3 * sizeof(unsigned char), NULL, &ret);
    cl_mem RGB_copy_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             imageSize * 3 * sizeof(unsigned char), NULL, &ret);

    // write to gpu memory
    clEnqueueWriteBuffer(command_queue, gray_mem_obj, CL_FALSE, 0,
                         imageSize * sizeof(unsigned char), imageGray, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, RGB_mem_obj, CL_FALSE, 0,
                         imageSize * 3 * sizeof(unsigned char), imageRGB, 0, NULL, NULL);

    // start measuring time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // calculate
    //printf("calculating\n");
    for (i = 0; i < (globalWidth - DESIRED_WIDTH); i++) {
        /**
         * SOBEL
         */
        // group sizes
        local_size[0] = WORKGROUP_SIZE;
        local_size[1] = WORKGROUP_SIZE;
        num_groups[0] = (size_t) ceil((double) height / local_size[0]);
        num_groups[1] = (size_t) ceil((double) width / local_size[1]);
        global_size[0] = local_size[0] * num_groups[0];
        global_size[1] = local_size[1] * num_groups[1];

        // set kernel args
        ret = clSetKernelArg(sobelKernel, 0, sizeof(cl_mem), (void *) &gray_mem_obj);
        ret |= clSetKernelArg(sobelKernel, 1, sizeof(cl_mem), (void *) &energy_mem_obj);
        ret |= clSetKernelArg(sobelKernel, 2, sizeof(unsigned char) * (local_size[0]+2) * (local_size[1]+2), NULL); //cache local memory
        ret |= clSetKernelArg(sobelKernel, 3, sizeof(int), &width);
        ret |= clSetKernelArg(sobelKernel, 4, sizeof(int), &height);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, sobelKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

        // wait for sobel to finish
        clFinish(command_queue);

        if (DEBUG == 0 && i == 20) {
            auto *energy = (unsigned *) malloc(imageSize * sizeof(unsigned));
            ret = clEnqueueReadBuffer(command_queue, energy_mem_obj, CL_TRUE, 0,
                                      imageSize * sizeof(unsigned), energy, 0, NULL, NULL);
            saveUnsignedImage(energy, width, height, "images/sobel_gpu_image2.png");
            free(energy);
        }

        /**
         * CUMULATIVE
         */
        // group sizes
        localSize = WORKGROUP_SIZE;
        globalSize = nearestMultipleOf(width, WORKGROUP_SIZE);

        // calculation
        /*for (row = height - 2; row >= 0; row--) {
            // set kernel args
            ret = clSetKernelArg(cumulativeBasicKernel, 0, sizeof(cl_mem), &energy_mem_obj);
            ret |= clSetKernelArg(cumulativeBasicKernel, 1, sizeof(int), &width);
            ret |= clSetKernelArg(cumulativeBasicKernel, 2, sizeof(int), &height);
            ret |= clSetKernelArg(cumulativeBasicKernel, 3, sizeof(int), &row);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, cumulativeBasicKernel, 1, NULL,
                    &globalSize, &localSize, 0, NULL, NULL);

            // wait for kernel to finish
            clFinish(command_queue);
        }*/

        local_size[0] = 7;
        local_size[1] = WORKGROUP_SIZE;
        global_size[0] = 7;
        global_size[1] = nearestMultipleOf(width / 2, WORKGROUP_SIZE);

        size_t offset[2] = {0, WORKGROUP_SIZE};

        //printf("all set\n");

        for (row = nearestMultipleOf(height, 7) / 7 - 1; row >= 0; row--) {
            ret = clSetKernelArg(trapezoid2CumulativeKernel, 0, sizeof(cl_mem), &energy_mem_obj);
            ret |= clSetKernelArg(trapezoid2CumulativeKernel, 1, localSize * localSize * sizeof(unsigned), NULL);
            ret |= clSetKernelArg(trapezoid2CumulativeKernel, 2, sizeof(int), &width);
            ret |= clSetKernelArg(trapezoid2CumulativeKernel, 3, sizeof(int), &height);
            ret |= clSetKernelArg(trapezoid2CumulativeKernel, 4, sizeof(int), &row);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, trapezoid2CumulativeKernel, 2, NULL,
                                         global_size, local_size, 0, NULL, NULL);

            // wait for kernel to finish
            clFinish(command_queue);

            ret = clSetKernelArg(trapezoid1CumulativeKernel, 0, sizeof(cl_mem), &energy_mem_obj);
            ret |= clSetKernelArg(trapezoid1CumulativeKernel, 1, localSize * localSize * sizeof(unsigned), NULL);
            ret |= clSetKernelArg(trapezoid1CumulativeKernel, 2, sizeof(int), &width);
            ret |= clSetKernelArg(trapezoid1CumulativeKernel, 3, sizeof(int), &height);
            ret |= clSetKernelArg(trapezoid1CumulativeKernel, 4, sizeof(int), &row);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, trapezoid1CumulativeKernel, 2, NULL,
                                         global_size, local_size, 0, NULL, NULL);

            // wait for kernel to finish
            clFinish(command_queue);
        }

        if (i == 0) {
            auto *energy = (unsigned *) malloc(imageSize * sizeof(unsigned));
            ret = clEnqueueReadBuffer(command_queue, energy_mem_obj, CL_TRUE, 0,
                    imageSize * sizeof(unsigned), energy, 0, NULL, NULL);
            saveUnsignedImage(energy, width, height, "images/test.png");
            free(energy);
            return;
        }

        /**
         * FIND SEAM - Part 1
         * Find min in top row - REDUCTION
         */
        // group sizes
        localSize = WORKGROUP_SIZE;
        numGroups = (size_t) ceil((double) width / localSize);
        globalSize = localSize * numGroups;

        // allocate gpu memory
        cl_mem reduction_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  numGroups * sizeof(unsigned), NULL, &ret);
        cl_mem reduction_index_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                        numGroups * sizeof(int), NULL, &ret);

        // set kernel args
        ret = clSetKernelArg(findMinKernel, 0, sizeof(cl_mem), &energy_mem_obj);
        ret |= clSetKernelArg(findMinKernel, 1, sizeof(cl_mem), &reduction_mem_obj);
        ret |= clSetKernelArg(findMinKernel, 2, sizeof(cl_mem), &reduction_index_mem_obj);
        ret |= clSetKernelArg(findMinKernel, 3, sizeof(unsigned) * localSize, NULL);
        ret |= clSetKernelArg(findMinKernel, 4, sizeof(int) * localSize, NULL);
        ret |= clSetKernelArg(findMinKernel, 5, sizeof(int), &width);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, findMinKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

        // wait for kernel to finish
        clFinish(command_queue);

        /**
         * FIND SEAM - Part 2
         * Final step in reduction + Find seam to remove
         */
        // group sizes
        globalSize = (size_t) nearestPower((int) numGroups);

        // allocate gpu memory
        cl_mem backtrack_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  height * sizeof(int), NULL, &ret);

        // set kernel args
        ret = clSetKernelArg(findSeamKernel, 0, sizeof(cl_mem), &energy_mem_obj);
        ret |= clSetKernelArg(findSeamKernel, 1, sizeof(cl_mem), &reduction_mem_obj);
        ret |= clSetKernelArg(findSeamKernel, 2, sizeof(cl_mem), &reduction_index_mem_obj);
        ret |= clSetKernelArg(findSeamKernel, 3, sizeof(unsigned) * globalSize, NULL);
        ret |= clSetKernelArg(findSeamKernel, 4, sizeof(int) * globalSize, NULL);
        ret |= clSetKernelArg(findSeamKernel, 5, sizeof(unsigned), &numGroups);
        ret |= clSetKernelArg(findSeamKernel, 6, sizeof(int), &width);
        ret |= clSetKernelArg(findSeamKernel, 7, sizeof(int), &height);
        ret |= clSetKernelArg(findSeamKernel, 8, sizeof(cl_mem), &backtrack_mem_obj);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, findSeamKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

        // free gpu memory
        ret = clReleaseMemObject(reduction_mem_obj);
        ret = clReleaseMemObject(reduction_index_mem_obj);

        // wait for kernel to finish
        clFinish(command_queue);

        /*auto *backtrack = (int *) malloc(height * sizeof(unsigned));
        ret = clEnqueueReadBuffer(command_queue, backtrack_mem_obj, CL_TRUE, 0,
                height*sizeof(unsigned), backtrack, 0, NULL, NULL);
        for (int j = 0; j < height;j++) {
            printf("%d, ", backtrack[j]);
        }
        free(backtrack);
        return;*/

        /**
         * DELETE SEAM
         */
        // group sizes - same as sobel
        local_size[0] = WORKGROUP_SIZE;
        local_size[1] = WORKGROUP_SIZE;
        num_groups[0] = (size_t) ceil((double) height / local_size[0]);
        num_groups[1] = (size_t) ceil((double) width / local_size[1]);
        global_size[0] = local_size[0] * num_groups[0];
        global_size[1] = local_size[1] * num_groups[1];

        // set kernel args
        ret = clSetKernelArg(deleteSeamKernel, 0, sizeof(cl_mem), &gray_mem_obj);
        ret |= clSetKernelArg(deleteSeamKernel, 1, sizeof(cl_mem), &gray_copy_mem_obj);
        ret |= clSetKernelArg(deleteSeamKernel, 2, sizeof(cl_mem), &RGB_mem_obj);
        ret |= clSetKernelArg(deleteSeamKernel, 3, sizeof(cl_mem), &RGB_copy_mem_obj);
        ret |= clSetKernelArg(deleteSeamKernel, 4, sizeof(cl_mem), &backtrack_mem_obj);
        ret |= clSetKernelArg(deleteSeamKernel, 5, sizeof(int), &width);
        ret |= clSetKernelArg(deleteSeamKernel, 6, sizeof(int), &height);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, deleteSeamKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

        // wait for kernel to finish
        clFinish(command_queue);

        // remove backtrack from memory
        ret = clReleaseMemObject(backtrack_mem_obj);

        // reduce width
        width--;

        // copy new image to old one
        clEnqueueCopyBuffer(command_queue, gray_copy_mem_obj, gray_mem_obj, 0, 0,
                            width * height * sizeof(unsigned char), 0, NULL, NULL);
        clEnqueueCopyBuffer(command_queue, RGB_copy_mem_obj, RGB_mem_obj, 0, 0,
                            width * height * 3 * sizeof(unsigned char), 0, NULL, NULL);
        clFinish(command_queue);

        //printf("%d", width);
    }
    printf("rotating\n");
    if (DESIRED_WIDTH >= globalWidth) {
        clEnqueueCopyBuffer(command_queue, gray_mem_obj, gray_copy_mem_obj, 0, 0,
                            width * height * sizeof(unsigned char), 0, NULL, NULL);
        clEnqueueCopyBuffer(command_queue, RGB_mem_obj, RGB_copy_mem_obj, 0, 0,
                            width * height * 3 * sizeof(unsigned char), 0, NULL, NULL);
        clFinish(command_queue);
    }
    printf("resizing again\n");
    if (DESIRED_HEIGHT < globalHeight) {
        /**
         * IMAGE ROTATION
         */
        // group sizes
        local_size[0] = WORKGROUP_SIZE;
        local_size[1] = WORKGROUP_SIZE;
        global_size[0] = nearestMultipleOf(height, WORKGROUP_SIZE);
        global_size[1] = nearestMultipleOf(width, WORKGROUP_SIZE);

        // set kernel args
        ret = clSetKernelArg(transposeKernel, 0, sizeof(cl_mem), &gray_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 1, sizeof(cl_mem), &gray_copy_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 2, WORKGROUP_SIZE * WORKGROUP_SIZE * sizeof(unsigned char), NULL);
        ret |= clSetKernelArg(transposeKernel, 3, sizeof(cl_mem), &RGB_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 4, sizeof(cl_mem), &RGB_copy_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 5, WORKGROUP_SIZE * WORKGROUP_SIZE * 3 * sizeof(unsigned char), NULL);
        ret |= clSetKernelArg(transposeKernel, 6, sizeof(int), &width);
        ret |= clSetKernelArg(transposeKernel, 7, sizeof(int), &height);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, transposeKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

        // wait for rotation to finish
        clFinish(command_queue);

        unsigned temp = height;
        height = width;
        width = temp;

        // calculate
        for (i = 0; i < (globalHeight - DESIRED_HEIGHT); i++) {
            /**
             * SOBEL
             */
            // group sizes
            local_size[0] = WORKGROUP_SIZE;
            local_size[1] = WORKGROUP_SIZE;
            num_groups[0] = (size_t) ceil((double) height / local_size[0]);
            num_groups[1] = (size_t) ceil((double) width / local_size[1]);
            global_size[0] = local_size[0] * num_groups[0];
            global_size[1] = local_size[1] * num_groups[1];

            // set kernel args
            ret = clSetKernelArg(sobelKernel, 0, sizeof(cl_mem), (void *) &gray_copy_mem_obj);
            ret |= clSetKernelArg(sobelKernel, 1, sizeof(cl_mem), (void *) &energy_mem_obj);
            ret |= clSetKernelArg(sobelKernel, 2, sizeof(unsigned char) * local_size[0] * local_size[1] +
                                                  (2 * local_size[0] + 2 * local_size[1]), NULL); //cache local memory
            ret |= clSetKernelArg(sobelKernel, 3, sizeof(int), &width);
            ret |= clSetKernelArg(sobelKernel, 4, sizeof(int), &height);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, sobelKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

            // wait for sobel to finish
            clFinish(command_queue);

            /**
             * CUMULATIVE
             */
            // group sizes
            localSize = WORKGROUP_SIZE;
            globalSize = nearestMultipleOf(width, WORKGROUP_SIZE);

            // calculation
            /*for (row = height - 2; row >= 0; row--) {
                // set kernel args
                ret = clSetKernelArg(cumulativeBasicKernel, 0, sizeof(cl_mem), (void *) &energy_mem_obj);
                ret |= clSetKernelArg(cumulativeBasicKernel, 1, sizeof(int), &width);
                ret |= clSetKernelArg(cumulativeBasicKernel, 2, sizeof(int), &height);
                ret |= clSetKernelArg(cumulativeBasicKernel, 3, sizeof(int), &row);

                // run kernel
                ret = clEnqueueNDRangeKernel(command_queue, cumulativeBasicKernel, 1, NULL,
                        &globalSize, &localSize, 0, NULL, NULL);

                // wait for kernel to finish
                clFinish(command_queue);
            }*/

            local_size[0] = 16;
            local_size[1] = WORKGROUP_SIZE;
            global_size[0] = 16;
            global_size[1] = nearestMultipleOf(width / 2, WORKGROUP_SIZE);

            size_t offset[2] = {0, WORKGROUP_SIZE};

            for (row = nearestMultipleOf(height, 16) / 16 - 1; row >= 0; row--) {
                ret = clSetKernelArg(trapezoid2CumulativeKernel, 0, sizeof(cl_mem), &energy_mem_obj);
                ret |= clSetKernelArg(trapezoid2CumulativeKernel, 1, localSize * localSize * sizeof(unsigned), NULL);
                ret |= clSetKernelArg(trapezoid2CumulativeKernel, 2, sizeof(int), &width);
                ret |= clSetKernelArg(trapezoid2CumulativeKernel, 3, sizeof(int), &height);
                ret |= clSetKernelArg(trapezoid2CumulativeKernel, 4, sizeof(int), &row);

                // run kernel
                ret = clEnqueueNDRangeKernel(command_queue, trapezoid2CumulativeKernel, 2, NULL,
                                             global_size, local_size, 0, NULL, NULL);

                // wait for kernel to finish
                clFinish(command_queue);

                ret = clSetKernelArg(trapezoid1CumulativeKernel, 0, sizeof(cl_mem), &energy_mem_obj);
                ret |= clSetKernelArg(trapezoid1CumulativeKernel, 1, localSize * localSize * sizeof(unsigned), NULL);
                ret |= clSetKernelArg(trapezoid1CumulativeKernel, 2, sizeof(int), &width);
                ret |= clSetKernelArg(trapezoid1CumulativeKernel, 3, sizeof(int), &height);
                ret |= clSetKernelArg(trapezoid1CumulativeKernel, 4, sizeof(int), &row);

                // run kernel
                ret = clEnqueueNDRangeKernel(command_queue, trapezoid1CumulativeKernel, 2, NULL,
                                             global_size, local_size, 0, NULL, NULL);

                // wait for kernel to finish
                clFinish(command_queue);
            }

            /*if (i == 0) {
                auto *energy = (unsigned *) malloc(imageSize * sizeof(unsigned));
                ret = clEnqueueReadBuffer(command_queue, energy_mem_obj, CL_TRUE, 0,
                        imageSize * sizeof(unsigned), energy, 0, NULL, NULL);
                saveUnsignedImage(energy, width, height, "../images/test.png");
                free(energy);
            }*/

            /**
             * FIND SEAM - Part 1
             * Find min in top row - REDUCTION
             */
            // group sizes
            localSize = WORKGROUP_SIZE;
            numGroups = (size_t) ceil((double) width / localSize);
            globalSize = localSize * numGroups;

            // allocate gpu memory
            cl_mem reduction_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                      numGroups * sizeof(unsigned), NULL, &ret);
            cl_mem reduction_index_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                            numGroups * sizeof(int), NULL, &ret);

            // set kernel args
            ret = clSetKernelArg(findMinKernel, 0, sizeof(cl_mem), &energy_mem_obj);
            ret |= clSetKernelArg(findMinKernel, 1, sizeof(cl_mem), &reduction_mem_obj);
            ret |= clSetKernelArg(findMinKernel, 2, sizeof(cl_mem), &reduction_index_mem_obj);
            ret |= clSetKernelArg(findMinKernel, 3, sizeof(unsigned) * localSize, NULL);
            ret |= clSetKernelArg(findMinKernel, 4, sizeof(int) * localSize, NULL);
            ret |= clSetKernelArg(findMinKernel, 5, sizeof(int), &width);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, findMinKernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

            // wait for kernel to finish
            clFinish(command_queue);

            /**
             * FIND SEAM - Part 2
             * Final step in reduction + Find seam to remove
             */
            // group sizes
            globalSize = (size_t) nearestPower((int) numGroups);

            // allocate gpu memory
            cl_mem backtrack_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                      height * sizeof(int), NULL, &ret);

            // set kernel args
            ret = clSetKernelArg(findSeamKernel, 0, sizeof(cl_mem), &energy_mem_obj);
            ret |= clSetKernelArg(findSeamKernel, 1, sizeof(cl_mem), &reduction_mem_obj);
            ret |= clSetKernelArg(findSeamKernel, 2, sizeof(cl_mem), &reduction_index_mem_obj);
            ret |= clSetKernelArg(findSeamKernel, 3, sizeof(unsigned) * globalSize, NULL);
            ret |= clSetKernelArg(findSeamKernel, 4, sizeof(int) * globalSize, NULL);
            ret |= clSetKernelArg(findSeamKernel, 5, sizeof(unsigned), &numGroups);
            ret |= clSetKernelArg(findSeamKernel, 6, sizeof(int), &width);
            ret |= clSetKernelArg(findSeamKernel, 7, sizeof(int), &height);
            ret |= clSetKernelArg(findSeamKernel, 8, sizeof(cl_mem), &backtrack_mem_obj);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, findSeamKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

            // free gpu memory
            ret = clReleaseMemObject(reduction_mem_obj);
            ret = clReleaseMemObject(reduction_index_mem_obj);

            // wait for kernel to finish
            clFinish(command_queue);

            if (DEBUG == 1 && i == 1) {
                auto *backtrack = (int *) malloc(height * sizeof(unsigned));
                ret = clEnqueueReadBuffer(command_queue, backtrack_mem_obj, CL_TRUE, 0,
                                          height * sizeof(unsigned), backtrack, 0, NULL, NULL);
                for (int j = 0; j < height; j++) {
                    printf("%d, ", backtrack[j]);
                }
                free(backtrack);
            }

            /**
             * DELETE SEAM
             */
            // group sizes - same as sobel
            local_size[0] = WORKGROUP_SIZE;
            local_size[1] = WORKGROUP_SIZE;
            num_groups[0] = (size_t) ceil((double) height / local_size[0]);
            num_groups[1] = (size_t) ceil((double) width / local_size[1]);
            global_size[0] = local_size[0] * num_groups[0];
            global_size[1] = local_size[1] * num_groups[1];

            // set kernel args
            ret = clSetKernelArg(deleteSeamKernel, 0, sizeof(cl_mem), &gray_copy_mem_obj);
            ret |= clSetKernelArg(deleteSeamKernel, 1, sizeof(cl_mem), &gray_mem_obj);
            ret |= clSetKernelArg(deleteSeamKernel, 2, sizeof(cl_mem), &RGB_copy_mem_obj);
            ret |= clSetKernelArg(deleteSeamKernel, 3, sizeof(cl_mem), &RGB_mem_obj);
            ret |= clSetKernelArg(deleteSeamKernel, 4, sizeof(cl_mem), &backtrack_mem_obj);
            ret |= clSetKernelArg(deleteSeamKernel, 5, sizeof(int), &width);
            ret |= clSetKernelArg(deleteSeamKernel, 6, sizeof(int), &height);

            // run kernel
            ret = clEnqueueNDRangeKernel(command_queue, deleteSeamKernel, 2, NULL, global_size, local_size, 0, NULL,
                                         NULL);

            // wait for kernel to finish
            clFinish(command_queue);

            // remove backtrack from memory
            clReleaseMemObject(backtrack_mem_obj);

            // reduce width
            width--;

            // copy new image to old
            clEnqueueCopyBuffer(command_queue, gray_mem_obj, gray_copy_mem_obj, 0, 0,
                                width * height * sizeof(unsigned char), 0, NULL, NULL);
            clEnqueueCopyBuffer(command_queue, RGB_mem_obj, RGB_copy_mem_obj, 0, 0,
                                width * height * 3 * sizeof(unsigned char), 0, NULL, NULL);
            clFinish(command_queue);
        }

        //rotate back
        // group sizes
        // group sizes
        local_size[0] = WORKGROUP_SIZE;
        local_size[1] = WORKGROUP_SIZE;
        global_size[0] = nearestMultipleOf(height, WORKGROUP_SIZE);
        global_size[1] = nearestMultipleOf(width, WORKGROUP_SIZE);

        // set kernel args
        ret = clSetKernelArg(transposeKernel, 0, sizeof(cl_mem), &gray_copy_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 1, sizeof(cl_mem), &gray_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 2, WORKGROUP_SIZE * WORKGROUP_SIZE * sizeof(unsigned char), NULL);
        ret |= clSetKernelArg(transposeKernel, 3, sizeof(cl_mem), &RGB_copy_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 4, sizeof(cl_mem), &RGB_mem_obj);
        ret |= clSetKernelArg(transposeKernel, 5, WORKGROUP_SIZE * WORKGROUP_SIZE * 3 * sizeof(unsigned char), NULL);
        ret |= clSetKernelArg(transposeKernel, 6, sizeof(int), &width);
        ret |= clSetKernelArg(transposeKernel, 7, sizeof(int), &height);

        // run kernel
        ret = clEnqueueNDRangeKernel(command_queue, transposeKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

        temp = height;
        height = width;
        width = temp;
        imageSize = width * height;

    }

    // read gpu memory to host memory
    imageSize = width * height;
    if (DESIRED_HEIGHT < globalHeight) {
        ret = clEnqueueReadBuffer(command_queue, RGB_mem_obj, CL_TRUE, 0,
                                  imageSize * 3 * sizeof(unsigned char), imageRGB, 0, NULL, NULL);
    } else {
        ret = clEnqueueReadBuffer(command_queue, RGB_copy_mem_obj, CL_TRUE, 0,
                                  imageSize * 3 * sizeof(unsigned char), imageRGB, 0, NULL, NULL);
    }

    // stop measuring time
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // print results
    printf("GPU:\n");
    printf("Resized image from %dx%d to %dx%d.\n", globalWidth, globalHeight, width, height);
    printf("Calculation time: %f.\n\n", elapsed);

    // save resized image
    FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageRGB, width,
                                                            height, width * 3, 24, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_JPEG, imageOutBitmap, "images/gpu_cut_image.png", 0);
    FreeImage_Unload(imageOutBitmap);

    // cleanup
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(sobelKernel);
    ret = clReleaseKernel(cumulativeBasicKernel);
    ret = clReleaseKernel(findMinKernel);
    ret = clReleaseKernel(findSeamKernel);
    ret = clReleaseKernel(deleteSeamKernel);
    ret = clReleaseKernel(transposeKernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(energy_mem_obj);
    ret = clReleaseMemObject(gray_mem_obj);
    ret = clReleaseMemObject(gray_copy_mem_obj);
    ret = clReleaseMemObject(RGB_mem_obj);
    ret = clReleaseMemObject(RGB_copy_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // free memory
    free(imageGray);
    free(imageRGB);
}

void resizeImageSerial(const char *imagePath) {
    unsigned char *imageGray, *imageRGB;
    unsigned *energy;
    unsigned width, height, pitchGray, pitchRGB, imageSize, globalWidth, globalHeight;
    int *backtrack, i;
    double elapsed;
    struct timespec start{}, finish{};

    // run tests
    /*if (test() != 0) {
        return 1;
    }*/

    // image reading
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, imagePath, 0);
    FIBITMAP *imageBitmapGray = FreeImage_ConvertToGreyscale(imageBitmap);
    width = FreeImage_GetWidth(imageBitmapGray);
    height = FreeImage_GetHeight(imageBitmapGray);
    pitchGray = FreeImage_GetPitch(imageBitmapGray);
    pitchRGB = FreeImage_GetPitch(imageBitmap);
    imageSize = height * width;

    // save image size
    globalWidth = width;
    globalHeight = height;

    // memory allocation
    imageGray = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    imageRGB = (unsigned char *) malloc(3 * imageSize * sizeof(unsigned char));
    energy = (unsigned *) malloc(imageSize * sizeof(unsigned));

    // load image to memory (gray for sobel and rgb for carving)
    FreeImage_ConvertToRawBits(imageGray, imageBitmapGray, pitchGray, 8,
                               0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_ConvertToRawBits(imageRGB, imageBitmap, pitchRGB, 24,
                               0xFF, 0xFF, 0xFF, TRUE);

    // remove read image
    FreeImage_Unload(imageBitmapGray);
    FreeImage_Unload(imageBitmap);

    // start measuring time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // find and delete seams (width)
    printf("starting work resizing %d to %d \n", globalWidth, DESIRED_WIDTH);

    backtrack = (int *) malloc(height * sizeof(int));
    for (i = 0; i < (globalWidth - DESIRED_WIDTH); i++) {

        sobelCPU(imageGray, energy, width, height);
        cumulativeCPU(energy, width, height);
        findSeam(energy, backtrack, width, height);
        deleteSeam(imageGray, imageRGB, backtrack, width, height);
        width--;
    }
    free(backtrack);

    if (DESIRED_HEIGHT < globalHeight) {
        // rotate images
        FIBITMAP *rotatedGrayImage = FreeImage_ConvertFromRawBits(imageGray, width,
                                                                  height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);
        FIBITMAP *rotatedRBGImage = FreeImage_ConvertFromRawBits(imageRGB, width,
                                                                 height, width * 3, 24, 0xFF, 0xFF, 0xFF, TRUE);

        rotatedGrayImage = FreeImage_Rotate(rotatedGrayImage, 90, NULL);
        rotatedRBGImage = FreeImage_Rotate(rotatedRBGImage, 90, NULL);

        FreeImage_ConvertToRawBits(imageGray, rotatedGrayImage, height, 8,
                                   0xFF, 0xFF, 0xFF, TRUE);
        FreeImage_ConvertToRawBits(imageRGB, rotatedRBGImage, height * 3, 24,
                                   0xFF, 0xFF, 0xFF, TRUE);

        // find and delete seams (height)
        backtrack = (int *) malloc(width * sizeof(int));
        for (i = 0; i < (globalHeight - DESIRED_HEIGHT); i++) {
            sobelCPU(imageGray, energy, height, width);
            cumulativeCPU(energy, height, width);
            findSeam(energy, backtrack, height, width);
            deleteSeam(imageGray, imageRGB, backtrack, height, width);
            height--;
            printf("seam deleted \n");
        }
        free(backtrack);

        // rotate image back and save it
        FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageRGB, height,
                                                                width, height * 3, 24, 0xFF, 0xFF, 0xFF, TRUE);
        imageOutBitmap = FreeImage_Rotate(imageOutBitmap, -90, NULL);
        FreeImage_Save(FIF_PNG, imageOutBitmap, "../images/cpu_cut_image.png", 0);
        FreeImage_Unload(imageOutBitmap);
    } else {
        FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageRGB, width,
                                                                height, width * 3, 24, 0xFF, 0xFF, 0xFF, TRUE);
        FreeImage_Save(FIF_PNG, imageOutBitmap, "../images/cpu_cut_image.png", 0);
        FreeImage_Unload(imageOutBitmap);
    }

    // stop
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // print results
    printf("CPU:\n");
    printf("Resized image from %dx%d to %dx%d.\n", globalWidth, globalHeight, width, height);
    printf("Calculation time: %f.\n", elapsed);

    // free memory
    free(imageGray);
    free(imageRGB);
    free(energy);
}
