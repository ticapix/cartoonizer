#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <vector>
#include <array>
#include <tuple>
#include <math.h>
#include <functional>

#include <potracelib.h>

#define LOG(preffix, fmt, ...) printf(preffix fmt "\n", __VA_ARGS__);
#define LOG_ERROR(fmt, ...) LOG("EE ", fmt, __VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG("   ", fmt, __VA_ARGS__)

#define WIN_MAIN "webcam view"
#define WIN_ANALYZE "analyze"
#define WIN_DEBUG "debug"


typedef enum result {
	OK,
	ERROR_MORE_THAN_ONE_CHANNEL,
} result;

static result last_error;

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

void for_each_pixel(cv::Mat &image, std::function<void(uchar * const pixel, int channels)> fn) {
	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();

	if (image.isContinuous()) {
		cols = cols * rows;
		rows = 1;
	}

	for (int j = 0; j < rows; ++j) {
		auto pixel = image.ptr(j);
		for (int i = 0; i < cols; ++i, pixel += channels)
			fn(pixel, channels);
	}
}

void increase_colour_saturation(cv::Mat &image, uchar inc) {
	cv::Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);

	for_each_pixel(hsv, [&inc](uchar * const pixel, int /*channels*/) {
		if (pixel[1] <= 255-inc)
			pixel[1] += inc;
		else
			pixel[1] = 255;
	});

	cvtColor(hsv, image, CV_HSV2BGR);
}


potrace_bitmap_t* create_bm_from_Mat(cv::Mat img) {
	if (img.channels() != 1) {
		last_error = ERROR_MORE_THAN_ONE_CHANNEL;
		return nullptr;
	}
	potrace_bitmap_t* bm = new potrace_bitmap_t();
	int word_size = sizeof(potrace_word) * 8;
	bm->w = img.cols;
	bm->h = img.rows;
	bm->dy = std::ceil(img.cols / word_size);
	bm->map = new potrace_word[bm->h * bm->dy];

	for (int j = 0; j < img.rows; ++j) {
		auto pixel = img.ptr(img.rows - 1 - j);
		for (int i = 0; i < img.cols; ++i) {
			if (pixel != 0) {
				auto word = (bm->map + j * bm->dy)[i / word_size];
				word |= 1 << (i % word_size);
			}
			pixel += 1; // bcz img.channels() = 1;
		}
	}
	return bm;
}


result release_bm(potrace_bitmap_t* bm) {
	delete[] bm->map;
	delete bm;
	bm = nullptr;
	return OK;
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


      cv::Mat img_source, img_gray, img_bw, img_edges, img_gray_blur;

      printf("using potrace: %s\n", potrace_version());



	  while (sourceCapture.isOpened()) //Show the image captured in the window and repeat
	    {
	      sourceCapture.read(img_source);
	      cvtColor(img_source, img_gray, CV_BGR2GRAY);
	      uchar med = 0;
	      assert(median(img_gray, med));
	      cv::threshold(img_gray, img_bw, med, 255, cv::THRESH_BINARY);

	      potrace_bitmap_t* bm = create_bm_from_Mat(img_bw);
	      assert(bm != nullptr);
	      potrace_param_t* param = potrace_param_default();
	      potrace_state_t *state = potrace_trace(param, bm);
	      assert(state->status == POTRACE_STATUS_OK);

	      potrace_path_t* path = state->plist;
	      std::vector<std::vector<cv::Point> > contours;
	      contours.empty();
	      while (path != nullptr) {
	    	  std::vector<cv::Point> contour;
	    	  for (int i = 0; i < path->curve.n; ++i) {
	    		  path->curve.tag[i];
	    		  contour.push_back(cv::Point(path->curve.c[i][2].x, path->curve.c[i][2].y));
	    	  }
	    	  contours.push_back(contour);
	    	  path = path->next;
	      }
	      printf("%d\n", contours.size());
	      cv::drawContours(img_source, contours, -1, cv::Scalar(0, 0, 0), 2);
	      potrace_state_free(state);
	      release_bm(bm);
		  potrace_param_free(param);

	      /// Convert the image to grayscale
	      cv::imshow(WIN_MAIN, img_source);
	      cv::imshow(WIN_DEBUG, img_bw);
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
