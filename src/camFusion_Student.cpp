
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes      
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }
        
    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{

    std::vector<cv::DMatch> matchesROI;
    // loop over all keypoint mathces, check for outliers and add keypoints that belong to roi
    for (auto it =  kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        
        // get correspoinding keypoints from previous and current frames
        cv::KeyPoint *currKeyPoint = &kptsCurr[it->trainIdx];

        if (boundingBox.roi.contains(currKeyPoint->pt))
        {
            matchesROI.push_back(*it);
        }
    }

    // compute mean and standard deviation of distances
    float distSum = 0.0, meanDist, standardDeviationDist = 0.0;   

    for(auto it = matchesROI.begin(); it != matchesROI.end(); ++it)
    {   
        distSum += it->distance;       
    }

    meanDist = distSum / matchesROI.size();
    
    for(auto it = matchesROI.begin(); it != matchesROI.end(); ++it)
    {   
        standardDeviationDist += pow(it->distance - meanDist, 2);
    }
    standardDeviationDist = sqrt(standardDeviationDist / matchesROI.size());

    for(auto it = matchesROI.begin(); it != matchesROI.end(); ++it)
    {   
        if (abs(it->distance - meanDist) < 3 * standardDeviationDist)
        {
            boundingBox.kptMatches.push_back(*it);
            boundingBox.keypoints.push_back(kptsCurr[it->trainIdx]);
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    /// Code from Lesson 3.3
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop
            if ((it1->trainIdx == it2->trainIdx) && (it1->queryIdx == it2->queryIdx))
            {
                continue;
            }
            double minDist = 50.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        
        return;
    }

    // compute camera-based TTC from distance ratios
    sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size()/2.0);
    double medianDistRatio = distRatios.size() % 2 == 0? (distRatios[medIndex+1]+ distRatios[medIndex]) / 2.0 : distRatios[medIndex+1];
    double dT = 1 / frameRate;

    if (medianDistRatio == 1)
    {
        TTC = -dT / 1e-04;
    }
    else
    {
        TTC = -dT / (1 - medianDistRatio);
    }
    
    //// End of Lesson 3.3 code
}

void getMinX(std::vector<LidarPoint> &lidarPoints, double &minX)
{
    minX = 1e09;
    float sum = 0.0, mean, standardDeviation = 0.0;   

    for(auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {   
        sum += it->x;       
    }

    mean = sum / lidarPoints.size();
    
    for(auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {   
        standardDeviation += pow(it->x - mean, 2);
    }
    standardDeviation = sqrt(standardDeviation / lidarPoints.size());

    for(auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {   
        if ((mean - it->x) > 3 * standardDeviation)
        {
            continue;
        }
        
        minX = minX > it->x? it->x : minX;
    }
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double minXPrev, minXCurr;
    getMinX(lidarPointsPrev, minXPrev);
    getMinX(lidarPointsCurr, minXCurr);

    TTC = minXCurr / (frameRate * (minXPrev - minXCurr));
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    vector<vector<int>> bbMatches;
    vector<vector<int>> bbMatchesCnt;
    // loop over all keypoint matches and find IDs of corresponding bounding boxes in previous and current frames
    for (int i = 0; i < matches.size(); ++i)
    {
        // get ID of each keypoint
        int trainID = matches[i].trainIdx;
        int queryID = matches[i].queryIdx;
  
        // get correspoinding keypoints from previous and current frames
        cv::KeyPoint *prevKeyPoint = &prevFrame.keypoints[queryID];
        cv::KeyPoint *currKeyPoint = &currFrame.keypoints[trainID];

        bool bPrevMatch = false, bCurrMatch = false;
        int prevBoxID, currBoxID;

        // match keypoint to a bounding box in previous frame
        for (auto it = prevFrame.boundingBoxes.begin(); it != prevFrame.boundingBoxes.end(); ++it)
        {
            if (it->roi.contains(prevKeyPoint->pt))
            {
                prevBoxID = it->boxID;
                bPrevMatch = true;
                break;
            }
        }
        if(!bPrevMatch)
        {
            continue;
        }
        // match keypoint to a bounding box in current frame
        for (auto it = currFrame.boundingBoxes.begin(); it != currFrame.boundingBoxes.end(); ++it)
        {
            if (it->roi.contains(currKeyPoint->pt))
            {
                currBoxID = it->boxID;
                bCurrMatch = true;
                break;
            }
        }
        if(!bCurrMatch)
        {
            continue;
        }
        // save pair of bounding boxes
        vector<int> pair{prevBoxID, currBoxID};
        bbMatches.push_back(pair);
    } 

    // Count number of matches for each pair of bounding boxes
    for (auto it1 = bbMatches.begin(); it1 != bbMatches.end(); ++it1)
    {
        
        int prevBoxID = (*it1)[0];
        int currBoxID = (*it1)[1];
        bool oldMatch = false;
        for (int i = 0; i < bbMatchesCnt.size(); ++i)
        {
            if (bbMatchesCnt[i][0] == prevBoxID && bbMatchesCnt[i][1] == currBoxID)
            {
                oldMatch = true;
                break;
            }
        }
        if (oldMatch)
        {
            continue;
        }
        int cnt = 1;
        for (auto it2 = bbMatches.begin(); it2 != bbMatches.end(); ++it2)
        {
            if (*it2 == *it1)
            {
                cnt++;
            }
        }
        vector<int> candidate{(*it1)[0], (*it1)[1], cnt};
        bbMatchesCnt.push_back(candidate);
    }
    
    // find best match for each bounding box from previous frame
    for (auto it1 = bbMatchesCnt.begin(); it1 != bbMatchesCnt.end(); ++it1)
    {
        vector<int> bestMatch{(*it1)[0], (*it1)[1],(*it1)[2]};
        
        auto it2 = bbMatchesCnt.begin();
        while (it2 != bbMatchesCnt.end())
        {   
            if ( ((*it2)[0] == bestMatch[0]) && ((*it2)[2] > bestMatch[2]) )
            {
                bestMatch[1] = (*it2)[1];
                bestMatch[2] = (*it2)[2];
            }         
            it2++;                      
        }
        
        auto find = bbBestMatches.find(bestMatch[0]);
        if(find != bbBestMatches.end())
        {
            find->second = bestMatch[1];
        }
        else
        {        
            bbBestMatches.insert({bestMatch[0], bestMatch[1]});
        }

    }
    
}
