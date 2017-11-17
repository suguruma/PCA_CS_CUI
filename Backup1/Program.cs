using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using KwsmLab.OpenCvSharp;
using KwsmLab.OpenCvSharp.Blob;
using KwsmLab.OpenCvSharp.Extern;
using KwsmLab.OpenCvSharp.MachineLearning;

namespace CVcs
{
    // 定数を定義する
    static class Constants
    {
        public const int LEARNING_DATA = 88;
        public const int EIGEN_DATA = 88; // LEARNING_DATA <= EIGEN_DATA であること
        public const int RECON_DATA = 88; // EIGEN_DATA <= RECON_DATA であること
        public const int MAXLENGTH = 256;
        public const int WHITE = 255;
        public const string RECON_NAME = "1.jpg";
    }

    class Program
    {
        static void Main(string[] args)
        {
            //=== 主成分分析 ===//
            //new Analysis.PrincipalComponent(); // PCA

            //--- Eigenfaces法 ---//
            //new Analysis.Eigenfaces(); // Grayscale
            //for (int i = 1; i <= Constants.LEARNING_DATA; i++)
            //{
            //int i = 3;
            //string RECON_NAME_NUM = i + ".jpg";
            //new Analysis.Reconstruct(RECON_NAME_NUM); // 再構成(Grayscale版)
            //}

            //new Analysis.EigenfacesRGB(); // RGB
            //for (int i = 1; i <= Constants.LEARNING_DATA; i++)
            //{
            //int i = 3;
            //string RECON_NAME_NUM = i + ".jpg";
            //new Analysis.ReconstructRGB(RECON_NAME_NUM); // 再構成(RGB版)
            //}

            int min = -1000, max = 1000;
            int step = 100;

            int eigen_num = 2;
            for (int i = min; i <= max; i = i + step)
            {
                new Analysis.OriginalProjectionRGB(eigen_num, i);
            }

            //=== 画像処理 ===//

            //--- RGB分割 ---//
            //new ImageProcessing.SplitRGB();





            // アプリケーション実行フォルダ: Application.StartupPath
            //Console.Write("Input Filename ->");
            //string filename = "../../Data/Images/" + Console.ReadLine(); // ユーザーの入力した文字列を1行読み込む

            //IplImage src = Cv.LoadImage(filename, LoadMode.GrayScale);
            //IplImage dst = Cv.CreateImage(new CvSize(src.Width, src.Height), BitDepth.U8, 1);
            //Cv.Canny(src, dst, 50, 200);
            //Cv.NamedWindow("src image");
            //Cv.ShowImage("src image", src);
            //Cv.NamedWindow("dst image");
            //Cv.ShowImage("dst image", dst);
            //Cv.WaitKey();
            //Cv.DestroyAllWindows();
            //Cv.ReleaseImage(src);
            //Cv.ReleaseImage(dst);
        }
    }
}
