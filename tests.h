//
// Created by amadej on 29. 12. 18.
//

#ifndef SEMINAR_TEST_H
#define SEMINAR_TEST_H

int test() {
    /**
    * Prepare test data.
    */
    unsigned width, height, imageSize;
    int *backtrack, i;

    width = 4;
    height = 4;
    imageSize = width * height;

    backtrack = (int *) malloc(height * sizeof(int));

    unsigned imageOut[] = {
            3,  4,  6,  2,
            4,  8,  1, 10,
            11,  2,  9,  5,
            3,  7,  6,  4
    };

    /**
    * Cumulatives test
    */
    unsigned char expectedOutput[] = {
            12, 10, 12,  8,
            9, 13,  6, 19,
            14,  5, 13,  9,
            3,  7,  6,  4
    };

    cumulativeCPU(imageOut, width, height);

    for (i = 0; i < imageSize; i++) {
        if (expectedOutput[i] != imageOut[i]) {
            printf("Error at index %d: %d (expected) != %d (got).\n Passed: 0/3\n",
                   i, expectedOutput[i], imageOut[i]);
            return 1;
        }
    }

    /**
    * Seam find and deletion test
    */
    unsigned char deleteExpectedOutput1[] = {
            12, 10, 12,
            9, 13, 19,
            14, 13,  9,
            7,  6,  4
    };
    int expectedBacktrack1[] = {
            3, 2, 1, 0
    };

    //findSeam(imageOut, backtrack, width, height);

    for (i = 0; i < height; i++) {
        if (expectedBacktrack1[i] != backtrack[i]) {
            printf("Error at index %d: %d (expected) != %d (got).\n Passed: 1/3\n",
                   i, expectedBacktrack1[i], backtrack[i]);
            return 1;
        }
    }

    deleteSeam(imageOut, backtrack, width, height);
    width--;
    imageSize = width * height;

    for (i = 0; i < imageSize; i++) {
        if (deleteExpectedOutput1[i] != imageOut[i]) {
            printf("Error at index %d: %d (expected) != %d (got).\n Passed: 1/3\n",
                   i, deleteExpectedOutput1[i], imageOut[i]);
            return 1;
        }
    }


    unsigned char deleteExpectedOutput2[] = {
            12, 12,
            13, 19,
            14,  9,
            7,  6
    };
    int expectedBacktrack2[] = {
            1, 0, 1, 2
    };

    //findSeam(imageOut, backtrack, width, height);

    for (i = 0; i < height; i++) {
        if (expectedBacktrack2[i] != backtrack[i]) {
            printf("Error at index %d: %d (expected) != %d (got).\n Passed: 2/3\n",
                   i, expectedBacktrack2[i], backtrack[i]);
            return 1;
        }
    }

    deleteSeam(imageOut, backtrack, width, height);
    width--;
    imageSize = width * height;

    for (i = 0; i < imageSize; i++) {
        if (deleteExpectedOutput2[i] != imageOut[i]) {
            printf("Error at index %d: %d (expected) != %d (got).\n Passed: 2/3\n",
                   i, deleteExpectedOutput2[i], imageOut[i]);
            return 1;
        }
    }

    /**
    * Cleanup
    */
    free(backtrack);

    printf("All tests passed (3/3).\n");
    return 0;
}

#endif //SEMINAR_TEST_H
