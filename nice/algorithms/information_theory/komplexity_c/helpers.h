#ifndef __HELPERS_H__
#define __HELPERS_H__


#define MAT3D(mat, x, y, z, xsize, ysize) (mat)[x + (y) * xsize + (z) * xsize * ysize]
#define MAT2D(mat, x, y, xsize) (mat)[x + (y) * xsize]


#endif
