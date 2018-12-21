#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// 参考：OpenCVとVisual C++による画像処理と認識（１３Ａ）
//  http://ishidate.my.coocan.jp/opencv_13A/opencv_13A.htm
int main(int argc, char *argv[]) {

  // --- 画像の読み込み ---
  if (argc != 2) {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }
  Mat src_image;
  src_image = imread(argv[1], 1);
  if (!src_image.data) {
    printf("No image data \n");
    return -1;
  }
  namedWindow("被検索画像", WINDOW_AUTOSIZE);
  imshow("被検索画像", src_image);
//  waitKey(0);
  // --- グレースケール画像の作成 ---
  Mat gray_image;
  cvtColor(src_image, gray_image, CV_BGR2GRAY);
  equalizeHist(gray_image, gray_image);
  namedWindow("グレー化画像", WINDOW_AUTOSIZE);
  imshow("グレー化画像", gray_image);
//  waitKey(0);
  //--- HOG特徴量の計算 ---
  // 被探索画像を64 x 128などの単位でスキャンし、
  // さらに、そこで、8 x 8などのセルを移動させて、HOG特徴の集まりを生成
  HOGDescriptor hog = HOGDescriptor(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9); // Default用
//  HOGDescriptor hog = HOGDescriptor(Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9); // Daimler 用

  //--- SVMにデータをセットする---
  // あらかじめ作成された検出用データを読み込んでSVM分類器を用意
  // getDefaultPeopleDetector():ウィンドウサイズが 64 x 128の分類器の学習済み係数、vector<float>型
   hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//  hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());


  // --- マルチスケール設定 ---
  // HOG特徴の集まりと検出用データを比較し、人であるか否かを判定
  vector<Rect> people;
  hog.detectMultiScale(gray_image, people, 0, Size(), Size(), 1.05, 2, false);

  //--- 検出画像描画 ---
  Mat output = src_image.clone();
  for (auto it = people.begin(); it != people.end(); ++it){
    rectangle(output, it->tl(), it->br(), Scalar(0, 255, 255), 2, 8, 0);
  }
  namedWindow("検出画像", WINDOW_AUTOSIZE);
  imshow("検出画像", output);
  waitKey(0);
  destroyAllWindows();
  return 0;
}
