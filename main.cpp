//
// Created by amadej on 29. 12. 18.
//

#include "header.h"

#define MAX_SOURCE_SIZE	16384
// wanted image size
#define DESIRED_WIDTH 400
#define DESIRED_HEIGHT 800
// debug mode
#define DEBUG 1

int main() {
    // Serial algorithm
    //resizeImageSerial();

    // Parallel algorithm
    resizeImageParallel();

    return 0;
}

void resizeImageParallel() {
    unsigned char *imageGray, *imageRGB;
    unsigned *energy;
    unsigned width, height, pitchGray, pitchRGB, imageSize, globalWidth, globalHeight;
    int *backtrack, row;
    double elapsed;
    struct timespec start{}, finish{};
    cl_int ret;

    // read image
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, "../images/image.jpg", 0);
    FIBITMAP *imageBitmapGray = FreeImage_ConvertToGreyscale(imageBitmap);
    width = FreeImage_GetWidth(imageBitmapGray);
    height = FreeImage_GetHeight(imageBitmapGray);
    pitchGray = FreeImage_GetPitch(imageBitmapGray);
    pitchRGB = FreeImage_GetPitch(imageBitmap);
    imageSize = height * width;

    // read and compile kernel
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("../kernel.cpp", "r");
    if (!fp)
    {
        fprintf(stderr, "Kernel file missing.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    source_str[source_size] = '\0';
    fclose(fp);

    // memory allocation
    imageGray = (unsigned char *) malloc(imageSize * sizeof(unsigned char));
    imageRGB = (unsigned char *) malloc(3 * imageSize * sizeof(unsigned char));
    energy = (unsigned *) malloc(imageSize * sizeof(unsigned));
    backtrack = (int *) malloc(height * sizeof(int));

    // opencl configuration
    cl_platform_id	platform_id[10];
    cl_uint			ret_num_platforms;
    char			buf[100];
    size_t			buf_len;
    ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);

    cl_device_id	device_id[10];
    cl_uint			ret_num_devices;
    cl_device_info device_info;
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
    printf("Local memory size: %lu\n", size[0]);

    cl_context context = clCreateContext(NULL, ret_num_devices, &device_id[0], NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // load image
    FreeImage_ConvertToRawBits(imageGray, imageBitmapGray, pitchGray, 8, 0xFF, 0xFF, 0xFF, TRUE);

    // prepare program
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source_str, NULL, &ret);

    // compile program
    ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);

    // logs
    size_t build_log_len;
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &build_log_len);
    char *build_log = (char*) malloc(sizeof(char)*(build_log_len + 1));
    ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
                                build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);

    // repeat
    /**
     * SOBEL
     */
    // allocate memory on gpu
    cl_mem input_image_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            imageSize * sizeof(unsigned char), NULL, &ret);
    cl_mem energy_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            imageSize * sizeof(unsigned), NULL, &ret);

    // write to gpu memory
    clEnqueueWriteBuffer(command_queue, input_image_mem_obj, CL_FALSE, 0,
            imageSize * sizeof(unsigned char), imageGray, 0, NULL, NULL);

    // prepare kernel
    cl_kernel kernel = clCreateKernel(program, "sobel", &ret);

    // calculate group sizes
    size_t local_size[2] = {(size_t)192, (size_t)3};
    size_t num_groups[2] = {(size_t)ceil((double)height/local_size[0]), (size_t)ceil((double)width/local_size[1])};
    size_t global_size[2] = {local_size[0] * num_groups[0], local_size[1] * num_groups[1]};

    // set kernel args
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_image_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&energy_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(unsigned char) * local_size[0] * local_size[1] +
            (2 * local_size[0] + 2 * local_size[1]), NULL); //cache local memory
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    ret |= clSetKernelArg(kernel, 4, sizeof(int), &height);

    // run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    // synchronously read result
    ret = clEnqueueReadBuffer(command_queue, energy_mem_obj, CL_TRUE, 0,
            imageSize*sizeof(unsigned), energy, 0, NULL, NULL);

    // save image if debug mode
    if (DEBUG) {
        const char *path = "../images/sobel_image.png";
        saveUnsignedImage(energy, width, height, path);
    }

    /**
     * CUMULATIVE
     */
    for (row = height - 2; row >= 0; row--) {
        // prepare kernel
        kernel = clCreateKernel(program, "cumulativeBasic", &ret);

        // set kernel args
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &energy_mem_obj);
        ret |= clSetKernelArg(kernel, 1, sizeof(int), &width);
        ret |= clSetKernelArg(kernel, 2, sizeof(int), &height);
        ret |= clSetKernelArg(kernel, 3, sizeof(int), &row);

        // run kernel
        auto globalSize = (size_t) width;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

        // wait for kernel to finish
        clFinish(command_queue);
    }

    // synchronously read result
    ret = clEnqueueReadBuffer(command_queue, energy_mem_obj, CL_TRUE, 0,
            imageSize*sizeof(unsigned), energy, 0, NULL, NULL);

    // save image if debug mode
    if (DEBUG) {
        const char *path = "../images/cumulative_image.png";
        saveUnsignedImage(energy, width, height, path);
    }

    /**
     * FIND SEAM
     */
    // find min in top row
    // group sizes
    size_t localSize = 32;
    auto numGroups = (size_t)ceil((double) width/localSize);
    size_t globalSize = localSize*numGroups;

    // gpu memory allocation
    cl_mem reduction_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            numGroups * sizeof(unsigned), NULL, &ret);
    cl_mem reduction_index_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            numGroups * sizeof(int), NULL, &ret);

    // prepare kernel
    kernel = clCreateKernel(program, "findMin", &ret);

    // set kernel args
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &energy_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &reduction_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &reduction_index_mem_obj);
    ret |= clSetKernelArg(kernel, 3, sizeof(unsigned) * localSize, NULL);
    ret |= clSetKernelArg(kernel, 4, sizeof(int) * localSize, NULL);
    ret |= clSetKernelArg(kernel, 5, sizeof(int), &width);

    // run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // wait for kernel to finish
    clFinish(command_queue);

    // finish reduction and find seam
    cl_mem backtrack_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            height * sizeof(int), NULL, &ret);

    // finish reduction and find seam to remove
    // prepare kernel
    kernel = clCreateKernel(program, "findSeam", &ret);

    // group sizes
    globalSize = (size_t) nearestPower((int) numGroups);

    // set kernel args
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &energy_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &reduction_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &reduction_index_mem_obj);
    ret |= clSetKernelArg(kernel, 3, sizeof(unsigned) * globalSize, NULL);
    ret |= clSetKernelArg(kernel, 4, sizeof(int) * globalSize, NULL);
    ret |= clSetKernelArg(kernel, 5, sizeof(unsigned), &numGroups);
    ret |= clSetKernelArg(kernel, 6, sizeof(int), &width);
    ret |= clSetKernelArg(kernel, 7, sizeof(int), &height);
    ret |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &backtrack_mem_obj);

    // run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // wait for kernel to finish
    clFinish(command_queue);

    /**
     * DELETE SEAM
     */
    // group sizes
    local_size[0] = 32; local_size[1] = 32;
    num_groups[0] = (size_t)ceil((double)height/local_size[0]);
    num_groups[1] = (size_t)ceil((double)width/local_size[1]);
    global_size[0] = local_size[0] * num_groups[0];
    global_size[1] = local_size[1] * num_groups[1];

    // allocate memory on gpu
    cl_mem gray_copy_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            imageSize * sizeof(unsigned char), NULL, &ret);
    cl_mem RGB_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            imageSize * 3 * sizeof(unsigned char), NULL, &ret);
    cl_mem RGB_copy_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            imageSize * 3 * sizeof(unsigned char), NULL, &ret);

    // load image
    FreeImage_ConvertToRawBits(imageRGB, imageBitmap, pitchRGB, 24, 0xFF, 0xFF, 0xFF, TRUE);

    // write RGB image to gpu memory
    clEnqueueWriteBuffer(command_queue, RGB_mem_obj, CL_FALSE, 0,
            imageSize * 3 * sizeof(unsigned char), imageRGB, 0, NULL, NULL);

    // prepare kernel
    kernel = clCreateKernel(program, "deleteSeam", &ret);

    // set kernel args
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &gray_copy_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &RGB_mem_obj);
    ret |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &RGB_copy_mem_obj);
    ret |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &backtrack_mem_obj);
    ret |= clSetKernelArg(kernel, 5, sizeof(int), &width);
    ret |= clSetKernelArg(kernel, 6, sizeof(int), &height);

    // run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    // wait for kernel to finish
    clFinish(command_queue);

    imageSize = (width-1)*height;
    ret = clEnqueueReadBuffer(command_queue, RGB_copy_mem_obj, CL_TRUE, 0,
            imageSize*3*sizeof(unsigned char), imageRGB, 0, NULL, NULL);


    FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageRGB, width-1,
            height, (width-1)*3, 24, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, imageOutBitmap, "../images/gpu_cut_image.png", 0);
    FreeImage_Unload(imageOutBitmap);

    // cleanup
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(energy_mem_obj);
    ret = clReleaseMemObject(backtrack_mem_obj);
    ret = clReleaseMemObject(reduction_mem_obj);
    ret = clReleaseMemObject(reduction_index_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // free memory
    free(imageGray);
    free(imageRGB);
    free(energy);
    free(backtrack);
}

void resizeImageSerial() {
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
    FIBITMAP *imageBitmap = FreeImage_Load(FIF_JPEG, "../images/image.jpg", 0);
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
    backtrack = (int *) malloc(height * sizeof(int));

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
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    // rotate image back and save it
    FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageRGB, height,
            width, height*3, 24, 0xFF, 0xFF, 0xFF, TRUE);
    imageOutBitmap = FreeImage_Rotate(imageOutBitmap, -90, NULL);
    FreeImage_Save(FIF_PNG, imageOutBitmap, "../images/cpu_cut_image.png", 0);
    FreeImage_Unload(imageOutBitmap);

    // print results
    printf("Resized image from %dx%d to %dx%d.\n", globalWidth, globalHeight, width, height);
    printf("Calculation time: %f.", elapsed);

    // free memory
    free(imageGray);
    free(imageRGB);
    free(energy);
    free(backtrack);
}
