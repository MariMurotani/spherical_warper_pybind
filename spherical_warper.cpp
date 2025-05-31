#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

namespace py = pybind11;
using namespace cv;

py::array_t<uint8_t> warp_spherical(
    py::array_t<uint8_t> input_image,
    float scale,
    std::vector<std::vector<float>> K_list,
    std::vector<std::vector<float>> R_list
) {
    py::buffer_info buf = input_image.request();

    int height = buf.shape[0];
    int width  = buf.shape[1];
    int channels = buf.shape[2];

    int cv_type;
    if (channels == 4) {
        cv_type = CV_8UC4;
    } else if (channels == 3) {
        cv_type = CV_8UC3;
    } else {
        throw std::runtime_error("Only 3 or 4 channel images are supported");
    }

    cv::Mat src(height, width, cv_type, static_cast<uchar*>(buf.ptr));

    // カメラ行列と回転行列を初期化
    cv::Mat K(3, 3, CV_32F);
    cv::Mat R(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            K.at<float>(i, j) = K_list[i][j];
            R.at<float>(i, j) = R_list[i][j];
        }

    cv::detail::SphericalWarper warper(scale);
    cv::Mat warped;

    warper.warp(src, K, R, INTER_LINEAR, BORDER_CONSTANT, warped);

    return py::array_t<uint8_t>(
        {warped.rows, warped.cols, channels},
        {static_cast<ssize_t>(warped.step[0]),
         static_cast<ssize_t>(warped.step[1]),
         static_cast<ssize_t>(1)},
        warped.data,
        py::capsule(warped.data, [](void *) {})  // メモリ管理をOpenCVに任せる
    );
}

PYBIND11_MODULE(spherical_warper, m) {
    m.def("warp_spherical", &warp_spherical, "Spherical projection supporting RGB and RGBA input");
}
