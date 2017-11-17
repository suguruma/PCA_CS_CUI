using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.ImageProcessing
{
    class SplitRGB
    {
        // グローバル宣言
        IplImage src_img;  // 画像
        string folder, savefolder, filename; // ファイル位置，名前
        int iWidth = 0, iHeight = 0;
        int numImage;

        // RGB分割 (メイン関数)
        public SplitRGB()
        {
            //===== 画像処理 =====//
            Console.Write("*** Image Processing of PCA ***\n");
            Console.Write("Please Input Folder Name -> ");

            // ユーザーの入力したフォルダを1行読み込む
            folder = "../../Data/Images/" + Console.ReadLine();
            savefolder = "../../Data/Images/";

            //----- 画像の読み込み&書き出し -----//
            ReadOneImage();
            Console.Write("Input & Write Images ... ");
            ReadWriteImage();
            Console.Write("OK\n");
        }

        // 画像サイズ決定
        public void ReadOneImage()
        {
            // 1枚画像を読み込む(サイズの決定)
            filename = folder + "/" + "1.jpg"; //指定した画像形式で読み込む
            src_img = Cv.LoadImage(filename, LoadMode.Color);

            // 行列の設定
            iWidth = src_img.Width;		// 画像の縦幅
            iHeight = src_img.Height;	// 画像の横幅
            numImage = Constants.LEARNING_DATA;
            Console.Write("Image Size [Width:" + iWidth + " Height:" + iHeight + "]\n"); // サイズ確認
        }

        // 画像読み込み
        public void ReadWriteImage()
        {
            using(IplImage dstR_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1))
            using(IplImage dstG_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1))
            using(IplImage dstB_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1))
            using(IplImage dst_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 3))
            {
                for (int i = 0; i < numImage; i++)
                {
                    // 入力ファイル
                    filename = folder + "/" + (i + 1) + ".jpg"; //指定した画像形式で読み込む
                    src_img = Cv.LoadImage(filename, LoadMode.Color);

                    // チャンネル分割(IplImage:B,G,R順)
                    Cv.Split(src_img, dstB_img, dstG_img, dstR_img, null);

                    filename = savefolder + "ImageR/" + String.Format("{0}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dstR_img);

                    filename = savefolder + "ImageG/" + String.Format("{0}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dstG_img);

                    filename = savefolder + "ImageB/" + String.Format("{0}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dstB_img);

                    // 結合(不要)
                    Cv.Merge(dstB_img, dstG_img, dstR_img, null, dst_img);
                    filename = savefolder + "Merge/" + String.Format("{0}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dst_img);
                }
            }
        }
    }
}
