/*
 * seam_carving.h
 *
 * Public API for the seam carving library.
 *
 * Seam carving is a content-aware image resizing technique that removes (or
 * inserts) paths of least visual importance — called "seams" — rather than
 * uniformly scaling the image.  The result preserves salient content such as
 * faces and edges far better than a standard bicubic or bilinear resize.
 *
 * Algorithm overview:
 *   1. Compute an "energy map": a scalar value per pixel representing how
 *      visually important that pixel is (high energy = important).
 *   2. Find the lowest-total-energy connected path from the top row to the
 *      bottom row — one pixel per row, moving at most one column left or right
 *      between adjacent rows.  This is the "minimum seam".
 *   3. Remove those pixels from the image.
 *   4. Repeat until the image reaches the desired width.
 *   Horizontal seams work identically on a 90°-rotated image.
 *
 * Six energy algorithms are supported (see EnergyAlgorithm below).
 * A standard bilinear resize (cv::resize) is also exposed for comparison.
 *
 * Dependencies: OpenCV 4.x  (core, imgcodecs, imgproc)
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// EnergyAlgorithm — selects how each pixel's importance is measured
// ---------------------------------------------------------------------------

/*
 * Each variant corresponds to a different edge-detection or gradient technique.
 * Higher energy values mean the pixel sits near a strong edge or texture, so
 * the algorithm avoids removing it.
 *
 *   SOBEL    — Classic Sobel operator.  Convolves with 3×3 horizontal and
 *              vertical gradient kernels, sums absolute responses across RGB.
 *              Good general-purpose choice; fast and robust.
 *
 *   PREWITT  — Similar to Sobel but uses equal weights in the kernel instead
 *              of the centre-weighted Sobel pattern.  Slightly noisier.
 *
 *   LAPLACIAN — Second-derivative operator.  Detects edges as zero-crossings.
 *               More sensitive to noise than first-derivative methods.
 *
 *   ROBERTS  — Diagonal gradient with 2×2 kernels.  Oldest edge detector;
 *              fast but misses near-horizontal and near-vertical edges.
 *
 *   CANNY    — Multi-stage detector with hysteresis thresholding.  Produces
 *              thin, clean edge maps.  Uses fixed thresholds 100/100.
 *
 *   FORWARD  — Forward energy from Rubinstein, Shamir & Avidan (2008).
 *              Instead of measuring existing gradients, it measures the new
 *              gradient that would be *introduced* by removing a seam.  Often
 *              produces fewer artefacts on images with gradual colour changes.
 */
enum class EnergyAlgorithm { SOBEL, PREWITT, LAPLACIAN, ROBERTS, CANNY, FORWARD };

/*
 * parseEnergyAlgorithm(code)
 *
 * Converts the single-letter CLI code into the corresponding enum value.
 *   "s" → SOBEL | "p" → PREWITT | "l" → LAPLACIAN
 *   "r" → ROBERTS | "c" → CANNY | "f" → FORWARD
 *
 * Throws std::invalid_argument for any unrecognised code.
 */
EnergyAlgorithm parseEnergyAlgorithm(const std::string& code);

// ---------------------------------------------------------------------------
// Core pipeline functions
// ---------------------------------------------------------------------------

/*
 * calculateEnergy(img, algo)
 *
 * Computes the energy map for the given BGR image using the chosen algorithm.
 *
 * Parameters:
 *   img  — Source image, CV_8UC3 (BGR, as returned by cv::imread).
 *   algo — Which edge-detection algorithm to use.
 *
 * Returns a CV_64F single-channel matrix of the same dimensions as img.
 * Each element holds the energy (importance) of the corresponding pixel.
 * Higher values mean the pixel is more important and should be preserved.
 */
cv::Mat calculateEnergy(const cv::Mat& img, EnergyAlgorithm algo);

/*
 * findSeam(energy)
 *
 * Finds the minimum-cost vertical seam through the energy map using
 * dynamic programming.
 *
 * A "vertical seam" is a connected path from row 0 to row H-1 where each
 * step moves to the same column, or one column left/right.  This gives
 * exactly one pixel per row.  The seam with the lowest total energy is the
 * one least likely to contain visible content.
 *
 * Algorithm:
 *   - M[0][j] = energy[0][j]  (seed the first row)
 *   - For each subsequent row i, M[i][j] = energy[i][j] + min(M[i-1][j-1..j+1])
 *   - backtrack[i][j] stores which column was chosen as parent for (i,j)
 *   - Trace back from the minimum of M[H-1] to recover the seam
 *
 * Parameters:
 *   energy — CV_64F energy map produced by calculateEnergy().
 *
 * Returns a std::vector<int> of length img.rows where element i is the
 * column index of the seam pixel in row i.
 */
std::vector<int> findSeam(const cv::Mat& energy);

/*
 * removeSeam(img, seam)
 *
 * Creates a new image with the seam pixels deleted, reducing width by 1.
 *
 * For each row i, the pixel at column seam[i] is skipped; all pixels to its
 * right are shifted one position left.  The original image is not modified.
 *
 * Parameters:
 *   img  — Source image (any type, any number of channels).
 *   seam — Column indices from findSeam(), one per row.
 *
 * Returns a new cv::Mat with the same height and type, but width = img.cols - 1.
 */
cv::Mat removeSeam(const cv::Mat& img, const std::vector<int>& seam);

/*
 * resizeImage(img, targetCols, algo)
 *
 * Iteratively removes seams until the image is targetCols columns wide.
 * This is the main resize loop: it calls calculateEnergy → findSeam →
 * removeSeam in sequence, (img.cols - targetCols) times.
 *
 * Progress is printed to stderr every 10 seams so the caller can see
 * that work is happening on large images.
 *
 * Parameters:
 *   img        — Source image, CV_8UC3 BGR.
 *   targetCols — Desired output width in pixels.  Must be < img.cols.
 *   algo       — Energy algorithm to use at each iteration.
 *
 * Returns the resized image as a new cv::Mat.
 */
cv::Mat resizeImage(const cv::Mat& img, int targetCols, EnergyAlgorithm algo);

// ---------------------------------------------------------------------------
// Options struct — bundles all parameters for the top-level entry points
// ---------------------------------------------------------------------------

/*
 * SeamCarvingOptions
 *
 * Aggregates every parameter needed to run a resize job.  Passed by const
 * reference to runSeamCarving() and runNormalResize() so both share the same
 * interface and main.cpp needs only one argument-parsing block.
 *
 * Fields:
 *   inputPath    — Absolute or relative path to the source image file.
 *   outputPath   — Path where the result PNG will be written.
 *   energy       — Energy algorithm (default: SOBEL).
 *   direction    — 'v' removes vertical seams → reduces width.
 *                  'h' removes horizontal seams → reduces height.
 *                  Horizontal mode works by rotating the image 90° CCW,
 *                  running the vertical algorithm, then rotating back 90° CW.
 *   targetWidth  — Desired output width in pixels (used when direction='v').
 *                  Mutually exclusive with scale and targetHeight.
 *   targetHeight — Desired output height in pixels (used when direction='h').
 *   scale        — Fractional target size, e.g. 0.5 = half the relevant
 *                  dimension.  Must be in (0.0, 1.0) exclusive.
 *   normalResize — Reserved flag; not currently used by the library
 *                  (main.cpp switches on modeStr instead).
 *
 * Exactly one of {scale, targetWidth, targetHeight} should be non-zero.
 */
struct SeamCarvingOptions {
    std::string inputPath;
    std::string outputPath;
    EnergyAlgorithm energy   = EnergyAlgorithm::SOBEL;
    char direction           = 'v';
    int  targetWidth         = 0;
    int  targetHeight        = 0;
    double scale             = 0.0;
    bool normalResize        = false;
};

/*
 * runSeamCarving(opts)
 *
 * Top-level entry point for content-aware seam-carving resize.
 *
 * Reads the image at opts.inputPath, removes seams according to opts,
 * and writes the result to opts.outputPath as a PNG.
 *
 * For horizontal direction:
 *   Rotates 90° CCW → runs vertical seam removal → rotates 90° CW back.
 *   This keeps the core algorithm always working on vertical seams.
 *
 * Throws std::runtime_error if the input cannot be opened or output cannot
 * be written.  Throws std::invalid_argument if the target dimension is out
 * of range (must be < original and > 0).
 */
void runSeamCarving(const SeamCarvingOptions& opts);

/*
 * runNormalResize(opts)
 *
 * Standard bilinear resize using cv::resize (INTER_LINEAR).
 * Produces exactly the same output dimensions as runSeamCarving() for the
 * same opts, so the web UI can show a direct side-by-side comparison.
 *
 * Unlike seam carving, this method squashes or stretches the entire image
 * uniformly — it has no notion of content importance.
 *
 * Throws std::runtime_error / std::invalid_argument in the same cases as
 * runSeamCarving().
 */
void runNormalResize(const SeamCarvingOptions& opts);
