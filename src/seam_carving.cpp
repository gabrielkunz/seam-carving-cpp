/*
 * seam_carving.cpp
 *
 * Implementation of the seam carving library declared in seam_carving.h.
 *
 * File layout:
 *   1. parseEnergyAlgorithm  — CLI code → enum
 *   2. applyKernelSumChannels — shared convolution helper used by all
 *                               backward-energy functions
 *   3. energySobel / energyPrewitt / energyLaplacian / energyRoberts
 *      energyCanny / energyForward  — six energy implementations
 *   4. calculateEnergy       — dispatcher
 *   5. findSeam              — dynamic programming seam search
 *   6. removeSeam            — pixel deletion
 *   7. resizeImage           — main resize loop
 *   8. runNormalResize        — standard cv::resize wrapper
 *   9. runSeamCarving         — top-level orchestrator
 */

#include "seam_carving.h"

#include <cfloat>       // DBL_MAX
#include <cmath>        // std::abs
#include <iostream>     // std::cerr for progress reporting
#include <stdexcept>    // std::runtime_error, std::invalid_argument

// ===========================================================================
// 1. Enum parsing
// ===========================================================================

/*
 * parseEnergyAlgorithm
 *
 * Maps the single-letter code that the user passes on the command line (or
 * that the Flask server forwards from the browser) to the internal enum.
 * Called once per process invocation in main.cpp.
 */
EnergyAlgorithm parseEnergyAlgorithm(const std::string& code) {
    if (code == "s") return EnergyAlgorithm::SOBEL;
    if (code == "p") return EnergyAlgorithm::PREWITT;
    if (code == "l") return EnergyAlgorithm::LAPLACIAN;
    if (code == "r") return EnergyAlgorithm::ROBERTS;
    if (code == "c") return EnergyAlgorithm::CANNY;
    if (code == "f") return EnergyAlgorithm::FORWARD;
    throw std::invalid_argument("Unknown energy algorithm code: " + code);
}

// ===========================================================================
// 2. Shared convolution helper
// ===========================================================================

/*
 * applyKernelSumChannels
 *
 * Applies a 2-D convolution kernel independently to each of the three BGR
 * colour channels of img_64f, takes the absolute value of each response,
 * and sums the three resulting matrices into a single-channel energy map.
 *
 * Why per-channel?
 *   A colour edge where only one channel changes is still an edge.  Summing
 *   the absolute responses across channels ensures such edges get high energy
 *   even if the average luminance barely changes.
 *
 * Why BORDER_REFLECT_101?
 *   This mirrors the image at the border WITHOUT repeating the edge pixel
 *   (e.g. ...c b | a b c...).  It matches the default behaviour of
 *   scipy.ndimage.convolve used in the original Python implementation.
 *
 * Parameters:
 *   img_64f — CV_64FC3 image (float64, 3 channels).  Must be pre-converted
 *             by the caller because filter2D on uint8 loses precision.
 *   kernel  — The convolution kernel (any size, CV_64F values).
 *   anchor  — Kernel anchor point.  Default (-1,-1) = kernel centre.
 *             Roberts uses (0,0) because its 2×2 kernel has no true centre.
 *
 * Returns a CV_64F single-channel matrix where each value is the sum of
 * absolute convolution responses across the three colour channels.
 */
static cv::Mat applyKernelSumChannels(const cv::Mat& img_64f,
                                      const cv::Mat& kernel,
                                      cv::Point anchor = cv::Point(-1, -1)) {
    // Split the 3-channel float image into three separate single-channel mats
    std::vector<cv::Mat> channels(3);
    cv::split(img_64f, channels);

    // Accumulator for the summed absolute responses
    cv::Mat result = cv::Mat::zeros(img_64f.rows, img_64f.cols, CV_64F);

    for (int c = 0; c < 3; ++c) {
        cv::Mat resp;
        // filter2D performs cross-correlation; for the symmetric/antisymmetric
        // kernels used here the result is equivalent to true convolution.
        cv::filter2D(channels[c], resp, CV_64F, kernel,
                     anchor, /*delta=*/0.0, cv::BORDER_REFLECT_101);

        // absdiff(x, 0) is the element-wise absolute value for a float mat
        cv::Mat absResp;
        cv::absdiff(resp, cv::Scalar(0), absResp);

        result += absResp;  // accumulate across channels
    }
    return result;
}

// ===========================================================================
// 3. Energy functions
// ===========================================================================

/*
 * energySobel
 *
 * Computes edge strength using the Sobel operator — a first-order gradient
 * detector.  Two 3×3 kernels capture horizontal (ksx) and vertical (ksy)
 * intensity changes:
 *
 *   ksx = [ 1  0 -1 ]    ksy = [ 1  2  1 ]
 *         [ 2  0 -2 ]          [ 0  0  0 ]
 *         [ 1  0 -1 ]          [-1 -2 -1 ]
 *
 * The centre row/column of each kernel is weighted 2× to reduce noise
 * sensitivity while still detecting edges.  The total energy is |gx| + |gy|
 * summed across RGB channels.
 */
static cv::Mat energySobel(const cv::Mat& img) {
    cv::Mat img64;
    img.convertTo(img64, CV_64FC3);  // convert to float64 before convolution

    // Horizontal gradient kernel — detects left/right edges
    cv::Mat ksx = (cv::Mat_<double>(3, 3) <<
        1,  0, -1,
        2,  0, -2,
        1,  0, -1);

    // Vertical gradient kernel — detects top/bottom edges
    cv::Mat ksy = (cv::Mat_<double>(3, 3) <<
        1,  2,  1,
        0,  0,  0,
       -1, -2, -1);

    return applyKernelSumChannels(img64, ksx) +
           applyKernelSumChannels(img64, ksy);
}

/*
 * energyPrewitt
 *
 * Similar to Sobel but with uniform weights (all 1s and -1s in the kernel).
 * Slightly less noise-resistant but computationally equivalent.
 *
 *   kpx = [ 1  0 -1 ]    kpy = [ 1  1  1 ]
 *         [ 1  0 -1 ]          [ 0  0  0 ]
 *         [ 1  0 -1 ]          [-1 -1 -1 ]
 */
static cv::Mat energyPrewitt(const cv::Mat& img) {
    cv::Mat img64;
    img.convertTo(img64, CV_64FC3);

    cv::Mat kpx = (cv::Mat_<double>(3, 3) <<
        1,  0, -1,
        1,  0, -1,
        1,  0, -1);

    cv::Mat kpy = (cv::Mat_<double>(3, 3) <<
        1,  1,  1,
        0,  0,  0,
       -1, -1, -1);

    return applyKernelSumChannels(img64, kpx) +
           applyKernelSumChannels(img64, kpy);
}

/*
 * energyLaplacian
 *
 * Second-order derivative detector — responds to rapid intensity changes in
 * any direction with a single 3×3 kernel:
 *
 *   kl = [ 0  1  0 ]
 *        [ 1 -4  1 ]
 *        [ 0  1  0 ]
 *
 * The -4 centre weight makes the kernel sum to 0, so flat regions produce
 * zero response.  More sensitive to noise than first-derivative methods;
 * works best on smooth images.
 */
static cv::Mat energyLaplacian(const cv::Mat& img) {
    cv::Mat img64;
    img.convertTo(img64, CV_64FC3);

    cv::Mat kl = (cv::Mat_<double>(3, 3) <<
        0,  1,  0,
        1, -4,  1,
        0,  1,  0);

    // Only one kernel needed (Laplacian is isotropic — responds equally in
    // all directions, so no separate X/Y pass is required)
    return applyKernelSumChannels(img64, kl);
}

/*
 * energyRoberts
 *
 * Diagonal gradient detector using the oldest edge operator (Roberts Cross,
 * 1963).  Two 2×2 kernels capture +45° and -45° gradients:
 *
 *   kr1 = [ 1  0 ]    kr2 = [ 0  1 ]
 *         [ 0 -1 ]          [-1  0 ]
 *
 * The anchor is set to (0,0) because a 2×2 kernel has no centre pixel.
 * Using (0,0) aligns the kernel with the top-left corner of the 2×2 window,
 * matching how scipy places the kernel in the original Python code.
 *
 * Note: Roberts misses near-horizontal and near-vertical edges; it works best
 * on images with predominantly diagonal features.
 */
static cv::Mat energyRoberts(const cv::Mat& img) {
    cv::Mat img64;
    img.convertTo(img64, CV_64FC3);

    cv::Mat kr1 = (cv::Mat_<double>(2, 2) <<
        1,  0,
        0, -1);

    cv::Mat kr2 = (cv::Mat_<double>(2, 2) <<
        0,  1,
       -1,  0);

    // Explicit (0,0) anchor to match scipy's 2×2 kernel placement behaviour
    cv::Point anchor(0, 0);
    return applyKernelSumChannels(img64, kr1, anchor) +
           applyKernelSumChannels(img64, kr2, anchor);
}

/*
 * energyCanny
 *
 * Uses OpenCV's full Canny edge detector, which performs:
 *   1. Gaussian smoothing (built-in, σ derived from the 3×3 aperture)
 *   2. Sobel gradient computation
 *   3. Non-maximum suppression (thins edges to single-pixel width)
 *   4. Hysteresis thresholding with thresholds 100 and 100
 *      (using L2 gradient norm — sqrt(gx²+gy²) instead of |gx|+|gy|)
 *
 * The result is a binary image (0 or 255) converted to float64.
 * Canny produces the cleanest, thinnest edges but its binary output means
 * all edges have equal weight; the seam algorithm treats them identically.
 */
static cv::Mat energyCanny(const cv::Mat& img) {
    // Canny requires a single-channel uint8 input
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    // thresholds: low=100, high=100  |  aperture=3  |  L2gradient=true
    cv::Canny(gray, edges, 100.0, 100.0, 3, true);

    // Convert uint8 binary edge map to float64 for consistency with other
    // energy functions (allows findSeam to treat all outputs identically)
    cv::Mat energy;
    edges.convertTo(energy, CV_64F);
    return energy;
}

/*
 * energyForward
 *
 * Implements "forward energy" from Rubinstein, Shamir & Avidan, SIGGRAPH 2008
 * ("Improved Seam Carving for Video Retargeting").
 *
 * Intuition:
 *   Backward energy (Sobel, Prewitt, etc.) measures the gradient that exists
 *   NOW.  Forward energy instead asks: "what new gradient would be CREATED if
 *   this seam were removed?"  Removing a seam brings previously non-adjacent
 *   pixels together; forward energy penalises seams that would introduce a
 *   large new discontinuity.
 *
 * Cost definitions (I = grayscale float64 image):
 *   cU[i][j] = |I[i][j+1] - I[i][j-1]|           (horizontal neighbour diff)
 *   cL[i][j] = |I[i-1][j] - I[i][j-1]| + cU[i][j] (cost of coming from upper-left)
 *   cR[i][j] = |I[i-1][j] - I[i][j+1]| + cU[i][j] (cost of coming from upper-right)
 *
 *   Wrap-around indexing (% W, % H) matches numpy's np.roll used in the
 *   original Python to keep array sizes constant during precomputation.
 *
 * Dynamic programming (m = cumulative cost matrix):
 *   Row 0: m[0][j] = 0
 *   Row i: m[i][j] = min(m[i-1][j-1]+cL, m[i-1][j]+cU, m[i-1][j+1]+cR)
 *   energy_map[i][j] = the cU / cL / cR term chosen by the argmin
 *
 * The returned energy_map is then fed into findSeam's own DP pass, which
 * finds the globally minimum path through it — exactly as the other
 * energy functions are used.
 */
static cv::Mat energyForward(const cv::Mat& img) {
    int H = img.rows, W = img.cols;

    // Convert to grayscale float64 — forward energy works on intensity only
    cv::Mat gray8;
    cv::cvtColor(img, gray8, cv::COLOR_BGR2GRAY);
    cv::Mat I;  // intensity matrix, shape (H, W), CV_64F
    gray8.convertTo(I, CV_64F);

    // --- Precompute the three cost maps ---
    // cU[i][j]: cost of the straight-up move from row i-1
    // cL[i][j]: cost of the left-diagonal move from row i-1
    // cR[i][j]: cost of the right-diagonal move from row i-1
    cv::Mat cU = cv::Mat::zeros(H, W, CV_64F);
    cv::Mat cL = cv::Mat::zeros(H, W, CV_64F);
    cv::Mat cR = cv::Mat::zeros(H, W, CV_64F);

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            // Wrap-around column/row indices (matches np.roll semantics)
            int jL = (j - 1 + W) % W;   // column to the left, wrapping
            int jR = (j + 1) % W;        // column to the right, wrapping
            int iU = (i - 1 + H) % H;   // row above, wrapping

            double cu = std::abs(I.at<double>(i, jR) - I.at<double>(i, jL));
            double cl = std::abs(I.at<double>(iU, j) - I.at<double>(i, jL)) + cu;
            double cr = std::abs(I.at<double>(iU, j) - I.at<double>(i, jR)) + cu;

            cU.at<double>(i, j) = cu;
            cL.at<double>(i, j) = cl;
            cR.at<double>(i, j) = cr;
        }
    }

    // --- Dynamic programming to compute cumulative costs ---
    cv::Mat m          = cv::Mat::zeros(H, W, CV_64F);  // cumulative cost
    cv::Mat energy_map = cv::Mat::zeros(H, W, CV_64F);  // per-cell insertion cost

    // Row 0 stays zero — no parent row to draw cost from

    for (int i = 1; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            int jL = (j - 1 + W) % W;
            int jR = (j + 1) % W;

            // Cumulative costs of the three possible parent pixels
            double mU = m.at<double>(i - 1, j);    // directly above
            double mL = m.at<double>(i - 1, jL);   // above-left
            double mR = m.at<double>(i - 1, jR);   // above-right

            // Add the insertion cost for each incoming direction
            double costU = mU + cU.at<double>(i, j);
            double costL = mL + cL.at<double>(i, j);
            double costR = mR + cR.at<double>(i, j);

            // Pick the cheapest parent and record its insertion cost
            if (costU <= costL && costU <= costR) {
                m.at<double>(i, j)          = costU;
                energy_map.at<double>(i, j) = cU.at<double>(i, j);
            } else if (costL <= costR) {
                m.at<double>(i, j)          = costL;
                energy_map.at<double>(i, j) = cL.at<double>(i, j);
            } else {
                m.at<double>(i, j)          = costR;
                energy_map.at<double>(i, j) = cR.at<double>(i, j);
            }
        }
    }

    // energy_map holds the per-pixel forward insertion costs.
    // findSeam will run its own DP on top of this map.
    return energy_map;
}

// ===========================================================================
// 4. Energy dispatcher
// ===========================================================================

/*
 * calculateEnergy
 *
 * Routes to the appropriate energy function based on the chosen algorithm.
 * This is the only function called by resizeImage; it hides the algorithm
 * selection from the rest of the pipeline.
 */
cv::Mat calculateEnergy(const cv::Mat& img, EnergyAlgorithm algo) {
    switch (algo) {
        case EnergyAlgorithm::SOBEL:     return energySobel(img);
        case EnergyAlgorithm::PREWITT:   return energyPrewitt(img);
        case EnergyAlgorithm::LAPLACIAN: return energyLaplacian(img);
        case EnergyAlgorithm::ROBERTS:   return energyRoberts(img);
        case EnergyAlgorithm::CANNY:     return energyCanny(img);
        case EnergyAlgorithm::FORWARD:   return energyForward(img);
    }
    throw std::invalid_argument("Unhandled energy algorithm");
}

// ===========================================================================
// 5. Seam finding — dynamic programming
// ===========================================================================

/*
 * findSeam
 *
 * Finds the minimum-cost vertical seam in an energy map using dynamic
 * programming.  A vertical seam visits exactly one pixel per row and
 * may move at most one column left or right between consecutive rows.
 *
 * Data structures:
 *   M          — CV_64F, same shape as energy.  M[i][j] = minimum total
 *                energy of any seam path from row 0 down to pixel (i, j).
 *                Initialised to energy (a copy) so row 0 is already set.
 *   backtrack  — CV_32S, same shape.  backtrack[i][j] = column index of
 *                the parent pixel chosen for (i, j).  Used during traceback
 *                to reconstruct the seam path.
 *
 * Fill phase (top to bottom):
 *   For each pixel (i, j), look at the three pixels directly above:
 *   columns max(j-1,0) to min(j+1, W-1).  Add the minimum of their M
 *   values to energy[i][j] and record which column was best.
 *
 * Traceback (bottom to top):
 *   Find the column with the smallest M value in the last row — that is the
 *   bottom of the cheapest seam.  Follow backtrack upward to reconstruct
 *   the full seam path row by row.
 */
std::vector<int> findSeam(const cv::Mat& energy) {
    int H = energy.rows, W = energy.cols;

    // M starts as a copy of the energy map; we fill it top-down
    cv::Mat M = energy.clone();

    // backtrack[i][j] = column of the parent chosen for pixel (i, j)
    cv::Mat backtrack(H, W, CV_32S, cv::Scalar(0));

    for (int i = 1; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            // Clamp to valid column range (handles left and right borders)
            int jMin = std::max(j - 1, 0);
            int jMax = std::min(j + 1, W - 1);

            // Find the cheapest parent in the row above
            int    bestCol = jMin;
            double bestVal = M.at<double>(i - 1, jMin);
            for (int k = jMin + 1; k <= jMax; ++k) {
                double v = M.at<double>(i - 1, k);
                if (v < bestVal) {
                    bestVal = v;
                    bestCol = k;
                }
            }

            backtrack.at<int>(i, j) = bestCol;          // record parent column
            M.at<double>(i, j)     += bestVal;           // accumulate cost
        }
    }

    // --- Traceback ---

    // Start at the minimum-cost cell in the last row
    std::vector<int> seam(H);
    double minVal = DBL_MAX;
    int    minCol = 0;
    for (int j = 0; j < W; ++j) {
        double v = M.at<double>(H - 1, j);
        if (v < minVal) { minVal = v; minCol = j; }
    }
    seam[H - 1] = minCol;

    // Walk upward row by row, following the backtrack pointers
    for (int i = H - 2; i >= 0; --i) {
        seam[i] = backtrack.at<int>(i + 1, seam[i + 1]);
    }

    return seam;  // seam[i] = column of the seam pixel in row i
}

// ===========================================================================
// 6. Seam removal
// ===========================================================================

/*
 * removeSeam
 *
 * Deletes the pixels identified by the seam vector from the image.
 * For each row i, everything to the left of seam[i] is copied as-is;
 * everything to the right is shifted one column left (the seam pixel itself
 * is simply skipped).
 *
 * The operation is non-destructive: a fresh cv::Mat is allocated and the
 * original image is left unchanged, which makes the resize loop simpler.
 */
cv::Mat removeSeam(const cv::Mat& img, const std::vector<int>& seam) {
    int H = img.rows, W = img.cols;

    // Output has the same height and type but is one column narrower
    cv::Mat output(H, W - 1, img.type());

    for (int i = 0; i < H; ++i) {
        int cut = seam[i];  // column to delete in this row

        // Copy pixels left of the cut (columns 0 .. cut-1)
        if (cut > 0)
            img.row(i).colRange(0, cut).copyTo(output.row(i).colRange(0, cut));

        // Copy pixels right of the cut (columns cut+1 .. W-1),
        // shifted one position left in the output
        if (cut < W - 1)
            img.row(i).colRange(cut + 1, W).copyTo(output.row(i).colRange(cut, W - 1));
    }

    return output;
}

// ===========================================================================
// 7. Resize loop
// ===========================================================================

/*
 * resizeImage
 *
 * Drives the core seam-carving loop.  At each iteration:
 *   1. Recompute the energy map for the current (shrinking) image.
 *      Re-computing every iteration is slower than caching but ensures
 *      accuracy — removing a seam changes the neighbourhood of many pixels.
 *   2. Find the minimum-cost seam via dynamic programming.
 *   3. Remove it, reducing the image width by 1.
 *
 * Progress is printed to stderr every 10 seams (using \r to overwrite the
 * same line) so long jobs remain observable without flooding the terminal.
 *
 * Parameters:
 *   img        — CV_8UC3 BGR source image.
 *   targetCols — Desired output width.  Must satisfy 0 < targetCols < img.cols.
 *   algo       — Energy algorithm used at each iteration.
 */
cv::Mat resizeImage(const cv::Mat& img, int targetCols, EnergyAlgorithm algo) {
    cv::Mat current = img.clone();           // working copy; original untouched
    int total = img.cols - targetCols;       // number of seams to remove

    for (int i = 0; i < total; ++i) {
        // Print progress every 10 seams; \r overwrites the current terminal line
        if (i % 10 == 0)
            std::cerr << "\rRemoving seam " << (i + 1) << " / " << total << std::flush;

        cv::Mat energy        = calculateEnergy(current, algo);  // step 1
        std::vector<int> seam = findSeam(energy);                // step 2
        current               = removeSeam(current, seam);       // step 3
    }

    // Print the final count (loop exits before printing the last multiple-of-10)
    if (total > 0)
        std::cerr << "\rRemoving seam " << total << " / " << total << "\n";

    return current;
}

// ===========================================================================
// 8. Standard resize (for side-by-side comparison)
// ===========================================================================

/*
 * runNormalResize
 *
 * Resizes the image with OpenCV's standard bilinear interpolation.
 * This is used by the web UI to generate the "Normal Resize" panel so users
 * can compare content-aware seam carving against a conventional resize that
 * uniformly squashes the entire image.
 *
 * Bilinear interpolation (INTER_LINEAR) weights each output pixel as the
 * weighted average of the four nearest source pixels.  It is fast and
 * produces smooth results but has no understanding of image content —
 * faces, horizons, and text will all be distorted equally.
 *
 * The output dimensions are determined by opts in exactly the same way as
 * runSeamCarving, so both functions always produce the same output size.
 */
void runNormalResize(const SeamCarvingOptions& opts) {
    cv::Mat img = cv::imread(opts.inputPath, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error("Cannot open image: " + opts.inputPath);

    // Start with original dimensions; only the relevant axis will change
    int targetW = img.cols, targetH = img.rows;

    if (opts.direction == 'h') {
        // Reducing height
        if (opts.scale > 0.0)            targetH = static_cast<int>(opts.scale * img.rows);
        else if (opts.targetHeight > 0)  targetH = opts.targetHeight;
        if (targetH <= 0 || targetH >= img.rows)
            throw std::invalid_argument("Target height out of range");
    } else {
        // Reducing width (default)
        if (opts.scale > 0.0)           targetW = static_cast<int>(opts.scale * img.cols);
        else if (opts.targetWidth > 0)  targetW = opts.targetWidth;
        if (targetW <= 0 || targetW >= img.cols)
            throw std::invalid_argument("Target width out of range");
    }

    cv::Mat out;
    // INTER_LINEAR = bilinear interpolation — smooth and general-purpose
    cv::resize(img, out, cv::Size(targetW, targetH), 0, 0, cv::INTER_LINEAR);

    if (!cv::imwrite(opts.outputPath, out))
        throw std::runtime_error("Cannot write output: " + opts.outputPath);
}

// ===========================================================================
// 9. Top-level seam-carving orchestrator
// ===========================================================================

/*
 * runSeamCarving
 *
 * End-to-end entry point: reads the source image, removes seams in the
 * requested direction, writes the result.
 *
 * Horizontal seams (reducing height):
 *   The core algorithm only removes vertical seams (one pixel per column).
 *   To remove horizontal seams we rotate the image 90° counter-clockwise,
 *   which swaps rows and columns, run the vertical algorithm, then rotate
 *   90° clockwise to restore the original orientation.
 *
 *   Rotation mapping:
 *     cv::ROTATE_90_COUNTERCLOCKWISE  ↔  numpy.rot90(img, 1, (0,1))
 *     cv::ROTATE_90_CLOCKWISE         ↔  numpy.rot90(img, 3, (0,1))
 *   (both match the original Python implementation exactly)
 *
 * Target dimension resolution (same logic for both directions):
 *   - If opts.scale > 0.0  → targetCols = scale × original dimension
 *   - Else if targetWidth/Height > 0  → use the explicit pixel count
 *   - Otherwise → invalid (validated by main.cpp before this is called)
 */
void runSeamCarving(const SeamCarvingOptions& opts) {
    cv::Mat img = cv::imread(opts.inputPath, cv::IMREAD_COLOR);
    if (img.empty())
        throw std::runtime_error("Cannot open image: " + opts.inputPath);

    if (opts.direction == 'h') {
        // --- Horizontal seam removal ---

        // Rotate 90° CCW: what was the image height becomes the working width
        cv::Mat rotated;
        cv::rotate(img, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);

        // The "column count" in the rotated frame corresponds to the original height
        int targetCols = 0;
        if (opts.scale > 0.0)
            targetCols = static_cast<int>(opts.scale * rotated.cols);
        else if (opts.targetHeight > 0)
            targetCols = opts.targetHeight;

        if (targetCols <= 0 || targetCols >= rotated.cols)
            throw std::invalid_argument("Target height out of range");

        cv::Mat resized = resizeImage(rotated, targetCols, opts.energy);

        // Rotate back 90° CW to restore the original orientation
        cv::rotate(resized, img, cv::ROTATE_90_CLOCKWISE);

    } else {
        // --- Vertical seam removal (default) ---

        int targetCols = 0;
        if (opts.scale > 0.0)
            targetCols = static_cast<int>(opts.scale * img.cols);
        else if (opts.targetWidth > 0)
            targetCols = opts.targetWidth;

        if (targetCols <= 0 || targetCols >= img.cols)
            throw std::invalid_argument("Target width out of range");

        img = resizeImage(img, targetCols, opts.energy);
    }

    if (!cv::imwrite(opts.outputPath, img))
        throw std::runtime_error("Cannot write output: " + opts.outputPath);
}
