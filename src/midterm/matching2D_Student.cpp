#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // opemcv workaround for binary descriptors
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        double maxRatio = 0.8; 
        vector<vector<cv::DMatch>> knnmatches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knnmatches, 2); // Finds the best 2 matches 
        t = ((double)cv::getTickCount() -t)/cv::getTickFrequency();
        cout << "knn matching took " << 1000 * t / 1.0 << endl;

        for (auto it = knnmatches.begin(); it != knnmatches.end(); ++it)
        {
            
            if ((*it)[0].distance < maxRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }

        }
        cout << "% of filtered matches = " << 100 * (knnmatches.size() - matches.size())/(double)knnmatches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();

    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        cerr << "unrecognized descriptor type!" << endl;        
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    int blockSize = 2;       // a block of size blockSize x blockSize is considered
    int apertureSize = 3;    // size of aperture for Sobel operator
    double k = 0.04;

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Non Maximum Suppression 
    double maxOverlap = 0.0;
    int minResponse = 100;

    for (int r = 0; r < dst_norm.rows; ++r)
    {
        for (int c = 0; c < dst_norm.cols; ++c)
        {
            int response = (int)dst_norm.at<float>(r,c);
            
            if (response > minResponse)
            {                   
                cv::KeyPoint newKeyPoint;               
                newKeyPoint.pt = cv::Point2f(c,r);
                newKeyPoint.size = 2 * apertureSize;

                if (!keypoints.empty())
                {
                    bool bOverlap = false;
                    for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                    {
                        double overlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                        
                        if (overlap > maxOverlap)
                        {
                            bOverlap = true;
                            if (newKeyPoint.response > it->response)
                            {
                                *it = newKeyPoint;
                                break;
                            }
                        }
                    }
                    if(!bOverlap)
                    {
                        keypoints.push_back(newKeyPoint);
                    }
                }
                else
                {
                    keypoints.push_back(newKeyPoint);
                }              
            }
        }
    } 
    t = ((double)cv::getTickCount()-t)/cv::getTickFrequency();
    cout << "HARRIS detected " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << "ms" << endl;
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // detector function for FAST, BRISK, ORB, AKAZE, SIFT

    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool nonMaxSuppression = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::DetectorType::TYPE_9_16;
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, nonMaxSuppression, type);
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cout << "FAST detected " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << "ms" << endl;
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cout << "BRISK detected " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << "ms" << endl;
    }
    else if (detectorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cout << "ORB detected " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << "ms" << endl;

    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cout << "AKAZE detected " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << "ms" << endl;
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        double t = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
        cout << "SIFT detected " << keypoints.size() << "keypoints in " << 1000 * t / 1.0 << "ms" << endl;

    }
    else 
    {
        cerr << "unrecognized detector type!" << endl;        
    }
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}