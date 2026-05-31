/*
 * main.cpp
 *
 * Command-line interface for the seam carving binary.
 *
 * This file is intentionally thin.  Its only job is to:
 *   1. Parse command-line arguments from argv into a SeamCarvingOptions struct.
 *   2. Validate that the arguments are consistent and in range.
 *   3. Dispatch to either runSeamCarving() or runNormalResize() depending on
 *      the --mode flag.
 *   4. Report elapsed wall-clock time to stderr.
 *
 * All image processing logic lives in seam_carving.cpp.
 *
 * Usage (also printed by --help / on bad arguments):
 *   ./seam_carving --input <path> --output <path>
 *                  [--mode seam|normal]   default: seam
 *                  [--energy s|p|l|r|c|f] default: s (Sobel)
 *                  [--direction v|h]      default: v (reduce width)
 *                  ( --scale 0.05–0.99 | --width <px> | --height <px> )
 *
 * The Flask web server (web/app.py) calls this binary as a subprocess and
 * passes the same flags, so the CLI contract is also the web API contract.
 */

#include "seam_carving.h"

#include <chrono>       // std::chrono for wall-clock timing
#include <cstring>      // std::strcmp for argument name matching
#include <iostream>     // std::cerr for error / progress messages
#include <stdexcept>    // std::exception for catch-all error handling
#include <string>

/*
 * printUsage
 *
 * Prints a concise usage summary to stderr and returns.
 * Called when argument count is wrong or an unknown flag is encountered.
 *
 * prog — argv[0], i.e. the name/path of the binary as invoked.
 */
static void printUsage(const char* prog) {
    std::cerr
        << "Usage: " << prog << "\n"
        << "  --input    <path>         Input image\n"
        << "  --output   <path>         Output image\n"
        << "  --mode     <seam|normal>  seam = seam carving (default), normal = standard resize\n"
        << "  --energy   <s|p|l|r|c|f> Energy algorithm (default: s, ignored for --mode normal)\n"
        << "               s = Sobel, p = Prewitt, l = Laplacian,\n"
        << "               r = Roberts, c = Canny, f = Forward\n"
        << "  --direction <v|h>         v = reduce width, h = reduce height (default: v)\n"
        << "  --scale    <0.05-0.99>    Target as fraction of original dimension\n"
        << "  --width    <px>           Target width in pixels  (use with --direction v)\n"
        << "  --height   <px>           Target height in pixels (use with --direction h)\n"
        << "  (exactly one of --scale, --width, --height must be given)\n";
}

int main(int argc, char* argv[]) {
    // Require at least one argument beyond the binary name
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // --- Argument storage ---

    SeamCarvingOptions opts;       // will be passed to the processing function

    std::string energyCode   = "s";    // default: Sobel
    std::string directionStr = "v";    // default: vertical (reduce width)
    std::string modeStr      = "seam"; // default: seam carving

    // Flags to detect whether the user supplied each resize argument,
    // so we can enforce the "exactly one of scale/width/height" rule
    bool hasScale = false, hasWidth = false, hasHeight = false;

    // --- Argument parsing loop ---

    for (int i = 1; i < argc; ++i) {
        /*
         * next(flag)
         *
         * Lambda that advances the loop counter and returns the next argv
         * element.  Used for flags that take a value: --input <path>, etc.
         * Exits immediately with an error message if no value follows.
         */
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << flag << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];  // advance i so the outer loop skips the value
        };

        if      (std::strcmp(argv[i], "--input")     == 0)  opts.inputPath  = next("--input");
        else if (std::strcmp(argv[i], "--output")    == 0)  opts.outputPath = next("--output");
        else if (std::strcmp(argv[i], "--mode")      == 0)  modeStr         = next("--mode");
        else if (std::strcmp(argv[i], "--energy")    == 0)  energyCode      = next("--energy");
        else if (std::strcmp(argv[i], "--direction") == 0)  directionStr    = next("--direction");
        else if (std::strcmp(argv[i], "--scale")     == 0) {
            opts.scale = std::stod(next("--scale"));   // parse as double
            hasScale   = true;
        }
        else if (std::strcmp(argv[i], "--width")  == 0) {
            opts.targetWidth = std::stoi(next("--width"));  // parse as int
            hasWidth         = true;
        }
        else if (std::strcmp(argv[i], "--height") == 0) {
            opts.targetHeight = std::stoi(next("--height"));
            hasHeight         = true;
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // --- Validation ---

    // Both --input and --output are mandatory
    if (opts.inputPath.empty() || opts.outputPath.empty()) {
        std::cerr << "Error: --input and --output are required\n";
        printUsage(argv[0]);
        return 1;
    }

    // Exactly one of --scale / --width / --height must be given
    int resizeArgCount = (int)hasScale + (int)hasWidth + (int)hasHeight;
    if (resizeArgCount != 1) {
        std::cerr << "Error: exactly one of --scale, --width, --height must be given\n";
        printUsage(argv[0]);
        return 1;
    }

    // Range checks for each resize argument
    if (hasScale && (opts.scale <= 0.0 || opts.scale >= 1.0)) {
        std::cerr << "Error: --scale must be between 0.0 and 1.0 (exclusive)\n";
        return 1;
    }
    if (hasWidth && opts.targetWidth < 1) {
        std::cerr << "Error: --width must be a positive integer\n";
        return 1;
    }
    if (hasHeight && opts.targetHeight < 1) {
        std::cerr << "Error: --height must be a positive integer\n";
        return 1;
    }

    // --direction must be "v" or "h"
    if (directionStr != "v" && directionStr != "h") {
        std::cerr << "Error: --direction must be 'v' or 'h'\n";
        return 1;
    }

    // --mode must be "seam" or "normal"
    if (modeStr != "seam" && modeStr != "normal") {
        std::cerr << "Error: --mode must be 'seam' or 'normal'\n";
        return 1;
    }

    // Store the direction character ('v' or 'h') in opts
    opts.direction = directionStr[0];

    // Convert the energy code string to the enum (throws on unknown code)
    try {
        opts.energy = parseEnergyAlgorithm(energyCode);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // --- Dispatch and timing ---

    // Record start time using a monotonic clock (unaffected by system time changes)
    auto t0 = std::chrono::steady_clock::now();

    try {
        if (modeStr == "normal")
            runNormalResize(opts);   // standard bilinear resize
        else
            runSeamCarving(opts);    // content-aware seam carving
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Report elapsed time (goes to stderr so it doesn't pollute stdout)
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "Done in " << elapsed << "s\n";

    return 0;  // success
}
