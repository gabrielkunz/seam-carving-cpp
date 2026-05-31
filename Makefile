CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

SRCS    := src/main.cpp src/seam_carving.cpp
TARGET  := build/seam_carving

# ---------------------------------------------------------------------------
# Auto-detect OpenCV
# ---------------------------------------------------------------------------
OPENCV_CFLAGS :=
OPENCV_LIBS   :=

# Try pkg-config first (works on most Linux installs and Homebrew with pkgconf)
PKG_CONFIG_RESULT := $(shell pkg-config --cflags --libs opencv4 2>/dev/null)
ifneq ($(PKG_CONFIG_RESULT),)
    OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
    OPENCV_LIBS   := $(shell pkg-config --libs   opencv4)
else
    # Fallback: Homebrew Apple Silicon
    ifneq ($(wildcard /opt/homebrew/include/opencv4/opencv2/core.hpp),)
        OPENCV_CFLAGS := -I/opt/homebrew/include/opencv4
        OPENCV_LIBS   := -L/opt/homebrew/lib \
                         -lopencv_core -lopencv_imgcodecs \
                         -lopencv_imgproc -lopencv_highgui
    # Fallback: Homebrew Intel
    else ifneq ($(wildcard /usr/local/include/opencv4/opencv2/core.hpp),)
        OPENCV_CFLAGS := -I/usr/local/include/opencv4
        OPENCV_LIBS   := -L/usr/local/lib \
                         -lopencv_core -lopencv_imgcodecs \
                         -lopencv_imgproc -lopencv_highgui
    # Fallback: Linux system install
    else ifneq ($(wildcard /usr/include/opencv4/opencv2/core.hpp),)
        OPENCV_CFLAGS := -I/usr/include/opencv4
        OPENCV_LIBS   := -lopencv_core -lopencv_imgcodecs \
                         -lopencv_imgproc -lopencv_highgui
    endif
endif

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: all clean run install-deps check-opencv

all: $(TARGET)

$(TARGET): $(SRCS) src/seam_carving.h | build
	$(CXX) $(CXXFLAGS) $(OPENCV_CFLAGS) -o $@ $(SRCS) $(OPENCV_LIBS)

build:
	mkdir -p build

clean:
	rm -rf build/

run: all
	cd web && FLASK_APP=app.py python3 -m flask run --host=0.0.0.0 --port=5000 --with-threads

install-deps:
	pip3 install -r requirements_web.txt

check-opencv:
	@echo "OPENCV_CFLAGS: $(OPENCV_CFLAGS)"
	@echo "OPENCV_LIBS:   $(OPENCV_LIBS)"
	@printf '#include <opencv2/opencv.hpp>\nint main(){cv::Mat m;return 0;}\n' \
	     | $(CXX) -std=c++17 $(OPENCV_CFLAGS) -x c++ - -o /dev/null $(OPENCV_LIBS) \
	     && echo "OpenCV: OK" \
	     || echo "OpenCV: FAILED — install with: brew install opencv"
