#include <bits/stdc++.h>
#define bug(x) cout << #x << " = " << (x) << '\n'
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <new_eigen.hpp>
#include <gp.h>
//#include <f2c.h>
//#include <clapack.h>

using namespace cv;
using namespace std;

using tensor = vector<Mat>;
using kernel = tensor;
const int Nt = 10;

inline bool check(int x, int y, int img_rows, int img_cols)
{
    if(x >= 0 and y >= 0 and x < img_cols and y < img_rows)
        return true;

    return false;
}

vector<pair<double , double >> trajetoria[Nt+2];

Rect2d cut_image(const Mat &frame)
{
    Mat _frame;
    frame.copyTo(_frame);
    putText(_frame, "Minimum size", Point(10, 10), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 255, 0));
    rectangle(_frame, Rect(20, 20, 21, 21), Scalar(255, 0, 0));
    Rect2d roi = selectROI("Tracker", _frame, false, false);

    return roi;
}

tensor get_features(const Mat &frame, const Rect2d &roi)
{
    Mat img = frame(roi), greyImg, d[4], aux(img.rows, img.cols, CV_64F),
            taux[3],
            desgraca1 = Mat::zeros(img.rows, img.cols, CV_64F),
            desgraca2 = Mat::zeros(img.rows, img.cols, CV_64F),
            corno;

    tensor features;
    split(img, taux);

    cvtColor(img, greyImg, CV_BGR2GRAY);
    Sobel(greyImg, d[0], CV_64F, 1, 0, 1);
    Sobel(greyImg, d[1], CV_64F, 0, 1, 1);
    Sobel(greyImg, d[2], CV_64F, 2, 0, 1);
    Sobel(greyImg, d[3], CV_64F, 0, 2, 1);

    //goodFeaturesToTrack(greyImg, corno, 500, 0.01, 10, Mat(), 3, 0, 0.04);


    for(int i = 0; i < 3; ++i)
        taux[i].convertTo(aux, CV_64F), features.push_back(aux), aux.release();
    for(int i = 0; i < 4; ++i)
        d[i].convertTo(aux, CV_64F), features.push_back(aux), aux.release();
    return features;
}

Mat get_cov(const tensor &roi_tensor)
{

    int m = roi_tensor[0].rows, n = roi_tensor[0].cols, sz = roi_tensor.size();
    Mat cov = Mat::zeros(sz, sz, CV_64F),
        mean = Mat::zeros(1, sz, CV_64F),
        aux = Mat::zeros(1, sz, CV_64F),
        aux2 = Mat::zeros(1, sz, CV_64F),
        integral_img;

    for(register uint i = 0; i < sz; ++i)
    {
        integral(roi_tensor[i], integral_img);
        mean.at<double>(0, i) = integral_img.at<double>(m, n) / (m * n);
    }
    for(register int i = 0;  i < m; ++i)
        for(register int j = 0; j < n; ++j)
        {
            for(register int k = 0; k < sz; ++k)
                aux.at<double>(0, k) = roi_tensor[k].at<double>(i, j);
            //bug(aux);
            mulTransposed(aux, aux2, true, mean, CV_64F);
            cov += aux2 / (m * n);
        }
    return cov;
}


const int INF = 1 << 30;
int main(int argc, char** argv)
{

    VideoCapture cap;
    if(argc == 2)
        cap = cv::VideoCapture(argv[1]);
    else if(argc == 3 and (std::string(argv[2]) == "-f"))
        cap = cv::VideoCapture(std::string(argv[1]) + "00000%3d.jpg");
    else
        return -1;
    if(!cap.isOpened())
        return -1;

    auto m = [](double) { return 0; };
    auto k = [](double x, double y) { return exp(-(x - y) * (x - y) / (2.0 * 1.00 * 1.00)); };
    GP gpx(m, k, 0.0);
    GP gpy(m, k, 0.0);


    Mat frame;
    int w_frame = cap.get(CV_CAP_PROP_FRAME_WIDTH), h_frame = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    Rect2d roi, r_track, r_track_aux;;
    VideoWriter  vw = VideoWriter("man.avi", CV_FOURCC('D', 'I', 'V', '3'), 20., Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
    /*if(argc < 2)
        for(int i = 0; i < 20; ++i) cap >> frame;
    cap >> frame;
    */
    tensor feat, track_feat, search_feat;
    vector<Mat> covs;
    deque<Mat> frame_covs;
    deque<Point> memo_track;
    Point p_aux, p_teste;
    Mat model, covalks, modelupdated = Mat::zeros(7, 7, CV_64F);
    int count = 0;
    char c;
    bool init = false, clean = false;
    init = true;
    for(;;)
    {

        cap >> frame;
        if(frame.empty()) break;

        resize(frame, frame, Size(w_frame, h_frame));
        if(init)
        {
            //woman = [42 x 108 from (199, 114)]
//            if(name.find("bolt") != string::npos)
//                roi = Rect2d(333, 150, 30, 70);
//            else
//                roi = selectROI("First", frame, false, false);

            //roi = Rect2d(Point(200.35,159.32), Point(245.48,113.74)); //bola
            roi = Rect2d(26,197,45,132);
            //feat = get_features(frame, Rect2d(0, 0, frame.cols, frame.rows));
            track_feat = get_features(frame, roi);
            rectangle(frame, roi, Scalar(0, 255, 0), 1);
            //circle(frame, Point2i(x, y), 1.5, Scalar(255, 0, 0), 2);
            //covs.push_back(get_cov(feat));
            covs.push_back(get_cov(track_feat));
            init = false;
            //continue;
        }
        if(clean) covs.clear(), feat.clear(), clean = false, cout << "tudo limpo\n";

        if(!track_feat.empty())
        {
            // w_frame - roi.width
            // h_frame - roi.height
            double dist = INF, aux;
//            for(uint i = 0; i < w_frame - roi.width; i += 15)
//                for(uint j = 0; j < h_frame - roi.height; j += 15)
//                {
//                    search_feat = get_features(frame, Rect2d(i, j, roi.width, roi.height));
//                    aux = adeus_eigen(covalks = get_cov(search_feat), covs[0]);
//                    if(aux < dist)
//                        dist = aux, r_track = Rect2d(i, j, roi.width, roi.height), model = covalks ;
//                    //cout << i << ' ' << j << '\n';
//                }
            if(memo_track.size() >= Nt){

            }

            Rect2d aux_ret;
            for(int i = -10; i <= 10; i+=2)
                for(int j = -10; j <= 10; j+=2)
                {
                    aux_ret = Rect2d(roi.x + j, roi.y + i, roi.width, roi.height);
                    if((aux_ret & Rect2d(0, 0, frame.cols, frame.rows)) == aux_ret)
                    {
                        search_feat = get_features(frame, aux_ret);
                        covalks = get_cov(search_feat);
                        aux = new_eigen::diss(covalks, covs[0]);
                        //bug(aux), bug(adeus_eigen(covalks, covs))

                        if(aux < dist)
                            dist = aux, r_track_aux = aux_ret, model = covalks ;
                    }
                }
            //bug(float(clock() - t) / CLOCKS_PER_SEC);
            //covs[0] = model;
            r_track = r_track_aux;
            if(dist == (INF)) {bug("morri");break;};
            rectangle(frame, r_track, Scalar(0, 0, 200), 2);

            cout << r_track.x+r_track.width*.5 << ", " << r_track.y+r_track.height*.5 << ", " << r_track.width << ", " << r_track.height << '\n';
            cout << (r_track.br()+r_track.tl())/2 << '\n';
            p_aux = (r_track.br()+r_track.tl())/2;

            gpx.push(count , p_aux.x);
            gpy.push(count, p_aux.y);
            memo_track.push_back(p_aux);


            frame_covs.push_back(model);

            if(frame_covs.size() == 20)
            {
                for(const auto &m: frame_covs)
                    modelupdated += m;
                modelupdated = modelupdated * 0.2;
                //bug(modelupdated);
                covs[0] = modelupdated;
                modelupdated.release();
                modelupdated = Mat::zeros(7, 7, CV_64F);
                frame_covs.pop_front();
                roi = r_track;
            }

        }
        count++;
        vw.write(frame);
        imshow("Tracking", frame);
//        //if(waitKey(30) == 27) break;
        c = (char) waitKey(1);
        if(c == 27) break;

        switch (c)
        {
        case 't':
            init = true;
            break;
         case 'c':
            clean = true;
        default:
            break;
        }

    }

    cap.release(), vw.release();
    return 0;
}
