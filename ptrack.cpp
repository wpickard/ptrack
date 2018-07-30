// This code is based on  the ssd_mobilenet_object_detection opencv demo
// and the RealSense2 library dnn example.
// Please see https://github.com/opencv/opencv/blob/master/LICENSE and
// https://github.com/IntelRealSense/librealsense/tree/master/wrappers/opencv/dnn

#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"

// Servo controller includes
#include "RPMSerialInterface.h"
#include <stdio.h>
#include <cmath>

// Timing help
#include <chrono>
#include <thread>

const size_t inWidth      = 300;
const size_t inHeight     = 300;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
//const float inScaleFactor = 1.0f
const float meanVal       = 127.5;
const char* classNames[]  = {"background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};

int main(int argc, char** argv) try
{
    using namespace cv;
    using namespace cv::dnn;
    using namespace rs2;
    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds

    Net net = readNetFromCaffe("MobileNetSSD_deploy.prototxt", 
                               "MobileNetSSD_deploy.caffemodel");

    //Configure Pololu Maestro servo controller
    const float pi = 3.141592653589793f;
    unsigned char deviceNumber = 12;    //Default Maestro device number
    unsigned char xChannelNumber = 0;    //Servo channel numbers?
    unsigned char yChannelNumber = 1;    //Servo channel numbers?
    const unsigned int channelMinValue = 500 * 4;
    const unsigned int channelMaxValue = 2500 * 4;
    const unsigned int channelValueRange = channelMaxValue - channelMinValue;
    const float channelMinAngle = 0;
    const float channelMaxAngle = pi;
    const float channelAngleRange = channelMaxAngle - channelMinAngle;
    const float ticksPerRadian = channelValueRange / channelAngleRange;
    std::string portName = "/dev/ttyACM0";
    unsigned int baudRate = 9600;
    
    printf("Creating serial interface '%s' at %d bauds\n", portName.c_str(), baudRate);
    std::string errorMessage;
    RPM::SerialInterface* serialInterface = RPM::SerialInterface::createSerialInterface( portName, baudRate, &errorMessage );
    if ( !serialInterface )
    {
	    printf("Failed to create serial interface. %s\n", errorMessage.c_str());
	    return -1;
    }

    // Initialize reference frame
    float xAngle = channelMinAngle + (channelAngleRange / 2);
    float yAngle = channelMinAngle + (channelAngleRange / 2);
    
    // Center camera to setup reference frame
    bool ret = false;
    ret = serialInterface->setTargetPP( deviceNumber, xChannelNumber, channelMinValue + (xAngle * ticksPerRadian) );
    ret = serialInterface->setTargetPP( deviceNumber, yChannelNumber, channelMinValue + (yAngle * ticksPerRadian) );

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);
    
    // Get camera intrinsics
    auto intrinsic = profile.get_intrinsics();
    auto principal_point = std::make_pair(intrinsic.ppx, intrinsic.ppy);
    auto focal_length = std::make_pair(intrinsic.fx, intrinsic.fy);
    rs2_distortion model = intrinsic.model;

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);
              
    printf("ticksPerRadian: %f\tchannelAngleRange: %f\tchannelValueRange: %d\tfx: %f\tfy: %f\twidth: %d\theight: %d\tppx: %f\tppy: %f\n", ticksPerRadian, channelAngleRange, channelValueRange, intrinsic.fx, intrinsic.fy, intrinsic.width, intrinsic.height, intrinsic.ppx, intrinsic.ppy);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    while (cvGetWindowHandle(window_name))
    {
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
        net.setInput(inputBlob, "data"); //set the network input
        Mat detection = net.forward("detection_out"); //compute output

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        // Crop both color and depth frames
        color_mat = color_mat(crop);
        depth_mat = depth_mat(crop);
        
        // Only user first person detection
        bool isFirstPerson = true;

        float confidenceThreshold = 0.8f;
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                object = object  & Rect(0, 0, depth_mat.cols, depth_mat.rows);

                // Calculate mean depth inside the detection region
                // This is a very naive way to estimate objects depth
                // but it is intended to demonstrate how one might 
                // use depht data in general
                Scalar m = mean(depth_mat(object));

                std::ostringstream ss;
                ss << classNames[objectClass] << " ";
                ss << std::setprecision(2) << m[0] << " meters away";
                String conf(ss.str());

                rectangle(color_mat, object, Scalar(0, 255, 0));
                int baseLine = 0;
                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                auto center = (object.br() + object.tl())*0.5;
                
                // Calculate new camera angle
                if(objectClass == 15 && isFirstPerson){
                    isFirstPerson = false;
                    
                    // Step-by-step calculation of new camera angle
                    signed detectedOffsetX = center.x - (color_mat.cols * 0.5);
                    signed detectedOffsetY = object.tl().y - (color_mat.rows * 0.25);
                    float detectedAngleX = atan2(detectedOffsetX, intrinsic.fx);
                    float detectedAngleY = atan2(detectedOffsetY, intrinsic.fy);
                    float desiredAngleX = xAngle + detectedAngleX * 0.25;
                    float desiredAngleY = yAngle + detectedAngleY * 0.25;
                    
                    // Clamp angle values
                    desiredAngleX = std::max(channelMinAngle, std::min(desiredAngleX, channelMaxAngle));
                    desiredAngleY = std::max(channelMinAngle, std::min(desiredAngleY, channelMaxAngle));
                    
                    // Calculate quarter millisecond "ticks" for servo pulse output
                    unsigned short positionX = channelMinValue + (desiredAngleX * ticksPerRadian);
                    unsigned short positionY = channelMinValue + (desiredAngleY * ticksPerRadian);
                    
                    // Move Camera
                    ret = serialInterface->setTargetPP( deviceNumber, xChannelNumber, positionX );
                    ret = serialInterface->setTargetPP( deviceNumber, yChannelNumber, positionY );
                    
                    printf("Person detected.\ndesiredAngleX: %f\tdesiredAngleY: %f\ndetectedAngleX: %+f\tdetectedAngleY: %+f\ncenter.x: %d\tcenter.y: %d\npositionX: %d\tpostitionY: %d\nwidth: %d\theight: %d\n", desiredAngleX, desiredAngleY, detectedAngleX, detectedAngleY, center.x, center.y, positionX, positionY, color_mat.cols, color_mat.rows);
                    
                    printf("Waiting for servo move to desiredAngle...");
                    bool areServosMoving = true;
                    ret = serialInterface->getMovingStatePP( deviceNumber, areServosMoving );
                	while ( areServosMoving )
	                {
		                printf(".");
		                sleep_for(milliseconds(10));
		                ret = serialInterface->getMovingStateCP( areServosMoving );
	                }
	                printf(" COMPLETE!\n");
	                
	                ret = serialInterface->getPositionPP( deviceNumber, 0, positionX );
	                
	                // Update reference frame
	                xAngle = desiredAngleX;
	                yAngle = desiredAngleY;
                }
                
                center.x = center.x - labelSize.width / 2;

                rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(255, 255, 255), CV_FILLED);
                putText(color_mat, ss.str(), center,
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        imshow(window_name, color_mat);
        if (waitKey(1) >= 0) break;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
