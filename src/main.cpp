#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <vector>
#include <array>
#include <tuple>
#include <math.h>

#define LOG(preffix, fmt, ...) printf(preffix fmt "\n", __VA_ARGS__);
#define LOG_ERROR(fmt, ...) LOG("EE ", fmt, __VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG("   ", fmt, __VA_ARGS__)

#define WIN_MAIN "webcam view"
#define WIN_ANALYZE "analyze"
#define WIN_DEBUG "debug"


bool hist(cv::Mat img, std::array<unsigned int, 256>& bins) {
	assert(img.type() == CV_8UC1);
	bins.fill(0);

	for(int row = 0; row < img.rows; ++row) {
	    uchar* p = img.ptr(row);
	    for(int col = 0; col < img.cols; ++col) {
	         ++bins[*p++];  //points to each pixel value in turn assuming a CV_8UC1 greyscale image
	    }
	}
	return true;
}

bool median(cv::Mat img, uchar& m) {
	assert(img.type() == CV_8UC1);
	static std::array<unsigned int, 256> bins;
	assert(hist(img, bins));

	unsigned long int sum = 0;
	unsigned long int mid = (img.rows * img.cols) / 2;
	for (int i = 0; i < 256; ++i) {
		sum += bins[i];
		if (sum > mid) {
			m = i;
			return true;
		}
	}
	return false;
}

int main(int argc, char **argv) {
	const std::string sourceInput = argv[1];
	  cv::VideoCapture sourceCapture(atoi(sourceInput.c_str()));
	  //  cv::VideoCapture sourceCapture(sourceInput);

	  if (!sourceCapture.isOpened())
	    {
	      LOG_ERROR("can't open %s", sourceInput.c_str());
	      return -1;
	    }
	  cv::namedWindow(WIN_MAIN);
	  cv::namedWindow(WIN_ANALYZE);
	  cv::namedWindow(WIN_DEBUG);

	  //  assert(checkBlackWhiteCross(generateMarkerTemplate(cv::Size(64, 64))));

      cv::Mat img_source, img_gray, img_edges, img_gray_blur;

      int canny_low = 255, canny_high = 255, erosion_size = 3, canny_ratio = 50;
      cv::createTrackbar("Canny Low", WIN_DEBUG, &canny_low, 512);
      cv::createTrackbar("Canny High", WIN_DEBUG, &canny_high, 512);
      cv::createTrackbar("Kernel size: 2n +1", WIN_DEBUG, &erosion_size, 11);
      cv::createTrackbar("Canny ratio", WIN_DEBUG, &canny_ratio, 100);

	  while (sourceCapture.isOpened()) //Show the image captured in the window and repeat
	    {
	      sourceCapture.read(img_source);

	      /// Convert the image to grayscale
	      cvtColor(img_source, img_gray, CV_BGR2GRAY);
	      uchar med = 0;
	      assert(median(img_gray, med));

	      cv::GaussianBlur(img_gray, img_gray_blur, cv::Size(0, 0), 3);
	      cv::addWeighted(img_gray, 1.5, img_gray_blur, -0.5, 0, img_gray);

//	      cv::Canny(img_gray, img_edges, canny_low, canny_high);
	      cv::Canny(img_gray, img_edges, med * ((0. + canny_ratio) / 100.), med * ((100. + canny_ratio) / 100.));


	      cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size));

//	      cv::dilate(img_edges, img_edges, element);

	      std::vector<std::vector<cv::Point> > contours;
          std::vector<cv::Vec4i> hierarchy;
          cv::findContours(img_edges, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

          int idx = 0;
          for( ; idx >= 0; idx = hierarchy[idx][0] )
          {
              cv::Scalar color(0); //rand()&255
              cv::drawContours(img_gray, contours, idx, color, CV_FILLED, 8, hierarchy);
          }

	      cv::imshow(WIN_ANALYZE, img_edges);
	      cv::imshow(WIN_MAIN, img_source);
	      cv::imshow(WIN_DEBUG, img_gray);
	      char c = cv::waitKey(33);
	      if (c == 27) {
	    	  cv::imwrite("/tmp/lastframe.jpg", img_source);
	    	  break;
	      }

	    }

	  cv::destroyWindow(WIN_MAIN);
	  cv::destroyWindow(WIN_ANALYZE);
	  cv::destroyWindow(WIN_DEBUG);
	  return 0;

}
