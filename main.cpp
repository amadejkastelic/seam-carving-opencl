//
// Created by amadej on 29. 12. 18.
//

#include "header.h"

#define MAX_SOURCE_SIZE	16384
// wanted image size
#define DESIRED_WIDTH 400
#define DESIRED_HEIGHT 800

int main() {
    // Serial algorithm
    // resizeImageSerial();

    // Parallel algorithm
    resizeImageParallel();

    return 0;
}

void resizeImageParallel() {
    unsigned char *imageGray, *imageRGB;
    unsigned *energy;
    unsigned width, height, pitchGray, pitchRGB, imageSize, globalWidth, globalHeight;
    int *backtrack, i;
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

    cl_ulong size;
    clGetDeviceInfo(device_id[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
    printf("Local memory size: %lu\n", size);

    cl_context context = clCreateContext(NULL, ret_num_devices, &device_id[0], NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);

    // load image
    FreeImage_ConvertToRawBits(imageGray, imageBitmapGray, pitchGray, 8, 0xFF, 0xFF, 0xFF, TRUE);

    // repeat
    /*
     * SOBEL
     */
    // allocate memory on gpu
    cl_mem image_input_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            imageSize * sizeof(unsigned char), NULL, &ret);
    cl_mem image_output_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            imageSize * sizeof(unsigned char), NULL, &ret);

    // write to gpu memory
    clEnqueueWriteBuffer(command_queue, image_input_mem_obj, CL_FALSE, 0,
            imageSize * sizeof(unsigned char), imageGray, 0, NULL, NULL);

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

    // prepare kernel
    cl_kernel kernel = clCreateKernel(program, "sobel", &ret);

    // calculate group sizes
    size_t local_size[2] = {(size_t)192, (size_t)3};
    size_t num_groups[2] = {(size_t)ceil((double)height/local_size[0]), (size_t)ceil((double)width/local_size[1])};
    size_t global_size[2] = {local_size[0] * num_groups[0], local_size[1] * num_groups[1]};

    // set kernel args
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_input_mem_obj);
    ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_output_mem_obj);
    ret |= clSetKernelArg(kernel, 2, sizeof(unsigned char) * local_size[0] * local_size[1] +
            (2 * local_size[0] + 2 * local_size[1]), NULL); //cache local memory
    ret |= clSetKernelArg(kernel, 3, sizeof(int), &width);
    ret |= clSetKernelArg(kernel, 4, sizeof(int), &height);

    // run kernel
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    // synchronously read result
    ret = clEnqueueReadBuffer(command_queue, image_output_mem_obj, CL_TRUE, 0,
            imageSize*sizeof(unsigned char), imageGray, 0, NULL, NULL);

    FIBITMAP *imageOutBitmap = FreeImage_ConvertFromRawBits(imageGray, width, height, pitchGray, 8, 0xFF, 0xFF, 0xFF, TRUE);
    FreeImage_Save(FIF_PNG, imageOutBitmap, "../images/sobel_image.png", 0);
    FreeImage_Unload(imageOutBitmap);

    //cumulative

    // find and delete seam

    // cleanup
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(image_input_mem_obj);
    ret = clReleaseMemObject(image_output_mem_obj);
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
