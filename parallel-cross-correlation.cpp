// parallel-cross-correlation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include "stb_image.h"
#include "stb_image_write.h"
typedef unsigned char Byte;

/*For all mathematical computations, a double typed array is used for the image data for utmost precision and sign preservation. 
But we read and write unsigned char data.*/
class Image {
public:
    /*Read an image of size Width x Height x channels*/
    static Image Read(const std::string& file_name) {
        int __width, __height, __channels;
        Byte* data = stbi_load(file_name.data(), &__width, &__height, &__channels, 0);

        if (data == nullptr) throw std::runtime_error(stbi_failure_reason());

        Image image(__width, __height, __channels);
        for (size_t i = 0; i < __width * __height * __channels; i++)
        {
            image._mem[i] = data[i];
        }
        delete[] data;
        return image;
    }
    Image(Image&& source) : width(source.width), height(source.height), channels(source.channels) {
        this->_mem = source._mem;
        source._mem = nullptr;
    }
    Image(const Image& source) : width(source.width), height(source.height), channels(source.channels) {
        size_t size = this->width * this->channels * this->height;
        this->_mem = new double[size]{ 0 };
        for (size_t i = 0; i < size; i++) this->_mem[i] = source._mem[i];
    }
    Image(size_t width, size_t height, size_t channels, double* mem = nullptr) : width(width), height(height), channels(channels){
        if (mem == nullptr)
            this->_mem = new double[width * height * channels]{ 0 };
        else this->_mem = mem;
    }

    double* get_mem() const{
        return this->_mem;
    }
    /*Write an image of any size and dimension*/
    void write(const std::string& file_name) {
        size_t size = this->width * this->channels * this->height;
        Byte* buff = new Byte[this->width * this->channels * this->height];
        /*Conversion is needed to write*/
        for (size_t i = 0; i < width * height * channels; i++) buff[i] = _mem[i];

        if (stbi_write_png(file_name.data(), this->width, this->height, this->channels, buff, this->channels * this->width * sizeof(Byte)) < 0) {
            throw std::runtime_error("Image write failed.");
        }
        delete[] buff;
    }
    ~Image() {
        delete[] this->_mem;
    }
    /*This operator assumes that the number of channels is 1. The entirety of this project uses two dimensional images.*/
    double& operator()(size_t i, size_t j) const {
        return *(this->_mem + (i * width + j));
    }
    const size_t width, height, channels;
private:
    double* _mem;
};

class Filter {
public:
    Filter(size_t width, size_t height) : width(width), height(height) {
        this->_mem = new double[width * height];
    }
    Filter(Filter&& source) : width(source.width), height(source.height) {
        this->_mem = source._mem;
        source._mem = nullptr;
    }
    Filter(const Filter& source) : width(source.width), height(source.height){
        size_t size = this->width * this->height;
        this->_mem = new double[size] { 0 };
        for (size_t i = 0; i < size; i++) this->_mem[i] = source._mem[i];
    }
    Filter(std::initializer_list<std::initializer_list<double>> init) : height(init.size()), width(init.begin()->size()) {
        this->_mem = new double[width * height];
        for (size_t i = 0; i < height; i++)
            for (size_t j = 0; j < width; j++)
                *(_mem + i * width + j) = *((init.begin() + i)->begin() + j);
    }
    double* get_mem() const{
        return this->_mem;
    }
    void print() {
        for (size_t i = 0; i < height; i++)
        {
            for (size_t j = 0; j < width; j++) {
                std::cout << (*this)(i, j) << ' ';
            }
            std::cout << '\n';
        }
    }
    double& operator()(size_t i, size_t j) const {
        return *(this->_mem + (i * width + j));
    }
    ~Filter() {
        delete[] this->_mem;
    }
    const size_t width, height;
private:
    double* _mem;
};


/*Writes in png format*/

/*The given image should have 3 channels(3d image). Outputs a grayscaled image which is now 2d.*/
Image gray_scale(const Image& source){
    if (source.channels != 3) throw std::logic_error("gray_scale() : Number of channels should be 3.");
    size_t num_pixels = source.width * source.height;
    
    double* buffer = new double[num_pixels];
    const double* source_mem = source.get_mem();
    Image to_return(source.width, source.height, source.channels);

    for (size_t i = 0; i < num_pixels; i++)
        buffer[i] = (source_mem[i * 3] + source_mem[i * 3 + 1] + source_mem[i * 3 + 2]) / 3;
        /* Averaging the rgb values*/

    return Image(source.width, source.height, 1, buffer);
}

/*The given image should have a single channel. This function cascades a value along along the specified boundaries with the value given.*/
Image pad_2D(const Image& source, double value, size_t up, size_t left, size_t right, size_t down) {
    if (source.channels != 1) throw std::logic_error("gray_scale() : Number of channels should be 3.");
    size_t new_width = source.width + left + right;
    size_t new_height = source.height + up + down;
    Image to_ret(new_width, new_height, 1);
    
    for (size_t i = 0; i < up; i++) /*Up*/
        for (size_t j = 0; j < new_width; j++)
            to_ret(i, j) = value;

    for (size_t i = new_height - down; i < new_height; i++) /*Down*/
        for (size_t j = 0; j < new_width; j++)
            to_ret(i, j) = value;
 
    for (size_t i = 0; i < left; i++) /*Left*/
        for (size_t j = 0; j < new_height; j++)
            to_ret(j, i) = value;

    for (size_t i = new_width - right; i < new_width; i++)/*Right*/
        for (size_t j = 0; j < new_height; j++)
            to_ret(j, i) = value;

    for (size_t i = up; i < up + source.height - 1; i++)
        for (size_t j = left; j < left + source.width - 1; j++)
            to_ret(i, j) = source(i - up, j - left);
    return to_ret;
}
/*The given image should have a single channel*/
Image correlate(const Image& source, const Filter& filter) {
    
    if (source.channels != 1) throw std::logic_error("gray_scale() : Number of channels should be 3.");

    size_t stride = 1, filter_width = filter.width, filter_height = filter.height, padding = 0, width = source.width, height = source.height;
    size_t out_width = ((width - filter_width + 2 * padding) / stride) + 1;
    size_t out_height = ((height - filter_height + 2 * padding) / stride) + 1;

    Image target(out_width, out_height, 1);

    for (size_t i = 0; i < out_height; i++) 
        for (size_t j = 0; j < out_width; j++)
            for (int a = 0; a < filter_height; a++)
                for (int b = 0; b < filter_width; b++)
                    target(i, j) += source(i + a, j + b) * filter(a, b);

    return target;
}

/*The given image should have a single channel*/
Image edge_detection(const Image& source) {

    if (source.channels != 1) throw std::logic_error("gray_scale() : Number of channels should be 3.");
    
    Filter sobelX = { {1, 0, -1},
                      {2, 0, -2},
                      {1, 0, -1} };
    Filter sobelY = {{ 1,  2,  1},
                     { 0,  0,  0},
                     {-1, -2, -1} };
    Image X = correlate(source, sobelX);
    Image Y = correlate(source, sobelY);

    /*Combining the vertical and horizontal edges*/
    for (size_t i = 0; i < X.height; i++) {
        for (size_t j = 0; j < X.width; j++)
        {
            X(i, j) = sqrt(pow(X(i, j), 2) + pow(Y(i, j), 2));
        }
    }
    return X;
}

/*The given image should have a single channel*/
Image gaussian_blurr(Image& source, double radius, double sigma) {
    if (source.channels != 1) throw std::logic_error("gray_scale() : Number of channels should be 3.");
    Filter gauss(radius, radius);
    double sum = 0;
    for (int i = 0; i < radius; i++) {
        for (int j = 0; j < radius; j++) {
            double sigma_pow_twice = 2 * sigma * sigma;
            gauss(i, j) = exp(-(pow(i - radius / 2.f, 2) + pow(j - radius / 2.f, 2)) / sigma_pow_twice) / (3.14159265358979323846 * sigma_pow_twice);
            sum += gauss(i, j);
        }
    }
    for (size_t i = 0; i < radius; i++)
        for (size_t j = 0; j < radius; j++)
            gauss(i, j) /= sum;
    return correlate(source, gauss);
}
int main()
{
    /* For edge detection, we first apply gray scale on the 3d image for to lose the 3rd dimension as we are not 
    concerned with the individual colors but only the intensity of the color. The gray scale operations is simply
    computing the average of each rgb pixel into one value. We apply a blur filter on the gray image to smooth out
    any extra small lines and roughness. This is achieved by correlating the gray image with the gaussian kernel. The 
    gaussian kernel has a complex formula which can be found over here.https://en.wikipedia.org/wiki/Gaussian_blur.
    The image is now ready for edge detection. We convolve the image with 2 filters for detecting both horizontal edges
    and vertical edges. They are called sobel fitlers. We then combine the 2 separate output edges into a single image by
    using the formula given here.https://en.wikipedia.org/wiki/Sobel_operator. And now we have all the edges.*/

    Image image = Image::Read("shapes.png");
    Image gray_scaled_image = gray_scale(image);
    Image blurred_image = gaussian_blurr(gray_scaled_image, 5, 1);
    Image edges = edge_detection(blurred_image);
    gray_scaled_image.write("gray.png");
    blurred_image.write("blur.png");
    edges.write("edges.png");
}

























//vector<vector<int>> padding2D(vector<vector<int>>& img, int ph, int pw) {
//    // padding ph lines of 0 on top and bottom, pw lines of 0 on left and right
//    int new_height = img.size() + 2 * ph;
//    int new_width = img[0].size() + 2 * pw;
//    vector<vector<int>> ret(new_height, vector<int>(new_width, 0));
//    /* Maybe this could be improved? */
//    for (int i = ph; i < ph + img.size(); i++)
//        for (int j = pw; j < pw + img[0].size(); j++)
//            ret[i][j] = img[i - ph][j - pw];
//    return ret;
//}
//vector < vect


/*
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
array = cv2.imread('download.jpg')
gray = np.mean(array, axis=2)

def get_gauss_kernel(radius, sigma):
  kernel = np.zeros((radius, radius))
  sum = 0
  for i in range(radius):
    for j in range(radius):
      sigma_pow_twice = 2 * sigma * sigma
      kernel[i, j] = np.exp(-(np.power(i - radius / 2, 2) + np.power(j - radius / 2, 2)) / sigma_pow_twice) / (3.14159265358979323846 * sigma_pow_twice)
      sum += kernel[i, j]
  for i in range(radius):
    for j in range(radius):
      kernel[i, j] /= sum

  return kernel

def correlate(image, kernel):
    stride = 1
    padding = 0
    out_width = int(((image.shape[1] - kernel.shape[1] + 2 * padding) / stride) + 1);
    out_height = int(((image.shape[0] - kernel.shape[0] + 2 * padding) / stride) + 1);
    output = np.zeros((out_height, out_width))
    for i in range(out_height):
      for j in range(out_width):
        for a in range(kernel.shape[0]):
          for b in range(kernel.shape[1]):
            output[i, j] += image[i + a, j + b] * kernel[a, b]
    return output

def gaussian_blur(image, radius, sigma):
  return correlate(image, get_gauss_kernel(radius, sigma))

def edge_detection(image):
  sobelX = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]], dtype='float64')
  sobelY = np.array([[ 1,  2,  1],
                     [ 0,  0,  0],
                     [-1, -2, -1]], dtype='float64')
  x = correlate(image, sobelX)
  cv2_imshow(x)
  y = correlate(image, sobelY)
  cv2_imshow(y)
  return np.sqrt(np.power(x, 2) + np.power(y, 2))

blurred = gaussian_blur(gray, 5, 5)
print(get_gauss_kernel(5, 5))
edges = edge_detection(blurred)
cv2_imshow(edges)

*/