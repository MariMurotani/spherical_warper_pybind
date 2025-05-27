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
    cv::Mat src(buf.shape[0], buf.shape[1], CV_8UC3, static_cast<uchar*>(buf.ptr));

    // カメラ行列と回転行列を初期化
    cv::Mat K(3, 3, CV_32F);
    cv::Mat R(3, 3, CV_32F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K.at<float>(i, j) = K_list[i][j];
            R.at<float>(i, j) = R_list[i][j];
        }
    }

    cv::detail::SphericalWarper warper(scale);

    cv::Mat warped;
    warper.warp(src, K, R, INTER_LINEAR, BORDER_CONSTANT, warped);

    return py::array_t<uint8_t>(
        {warped.rows, warped.cols, 3},
        {static_cast<ssize_t>(warped.step[0]),
         static_cast<ssize_t>(warped.step[1]),
         static_cast<ssize_t>(1)},
        warped.data,
        py::capsule(warped.data, [](void *) {})
    );
}

py::array_t<uint8_t> warp_equirectangular(
    py::array_t<uint8_t> input_image,
    int out_width,
    int out_height
) {
    py::buffer_info buf = input_image.request();
    cv::Mat src(buf.shape[0], buf.shape[1], CV_8UC3, (uchar *)buf.ptr);

    int in_h = src.rows;
    int in_w = src.cols;

    cv::Mat output(out_height, out_width, CV_8UC3);

    for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
            // 緯度 θ ∈ [-π/2, π/2], 経度 φ ∈ [-π, π]
            float theta = (float(M_PI) * (0.5f - float(y) / out_height));   // latitude
            float phi   = (2.0f * float(M_PI) * (float(x) / out_width - 0.5f)); // longitude

            // 球面 → カメラ前方投影面（Z = 1）
            float X = cos(theta) * sin(phi);
            float Y = sin(theta);
            float Z = cos(theta) * cos(phi);

            float u = 0.5f + X / (2 * Z + 1e-6f);
            float v = 0.5f - Y / (2 * Z + 1e-6f);

            int ix = std::min(std::max(int(u * in_w), 0), in_w - 1);
            int iy = std::min(std::max(int(v * in_h), 0), in_h - 1);

            output.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(iy, ix);
        }
    }

    return py::array_t<uint8_t>(
        {output.rows, output.cols, 3},
        {static_cast<ssize_t>(output.step[0]),
         static_cast<ssize_t>(output.step[1]),
         static_cast<ssize_t>(1)},
        output.data,
        py::capsule(output.data, [](void *) {})
    );
}

PYBIND11_MODULE(spherical_warper, m) {
    m.def("warp_spherical", &warp_spherical, "Spherical cutout using OpenCV");
    m.def("warp_equirectangular", &warp_equirectangular, "Full equirectangular projection");
}