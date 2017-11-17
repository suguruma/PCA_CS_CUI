using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.Analysis
{
    class Eigenfaces
    {
        // グローバル宣言
        IplImage src_img, dst_img;  // 画像
        CvMat src_mat;  // 原画像行列
        CvMat mean_mat, center_mat; // 平均&分散
        string folder, savefolder, filename; // ファイル位置，名前
        int iWidth = 0, iHeight = 0, maxSize;
        int numImage, inline = 1;

        // 主成分分析 (メイン関数)
        public Eigenfaces()
        {
            //===== 画像処理(PCA) =====//
            Console.Write("*** Image Processing of PCA ***\n");
            Console.Write("Please Input Folder Name -> ");

            // ユーザーの入力したフォルダを1行読み込む
            folder = "../../Data/Images/" + Console.ReadLine();
            savefolder = "../../Data/Savefile/";

            //----- 画像の読み込み -----//
            ReadOneImage();
            Console.Write("Input File ... ");
            ReadImage();
            Console.Write("OK\n");

            //----- 平均画像(平均顔) -----//
            Console.Write("Mean ... ");
            Mean();
            filename = savefolder + "Mean_mat.csv"; // 出力ファイル名(平均ベクトル)
            WriteMat(mean_mat, filename);
            Console.Write("OK\n");

            //----- 分散(センタリング) -----//
            Console.Write("Centering ... ");
            Centering();
            Console.Write("OK\n");

            // src_matと同じ大きさ
            using (CvMat evect_mat = Cv.CreateMat(numImage, numImage, MatrixType.F32C1))
            using (CvMat ori_evect_mat = Cv.CreateMat(maxSize, numImage, MatrixType.F32C1))
            using (CvMat pjt_matT = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
            {
                using (CvMat eval_mat = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
                {
                    using (CvMat vc_mat = Cv.CreateMat(numImage, numImage, MatrixType.F32C1))
                    {
                        //----- 分散共分散行列 -----//
                        Console.Write("Variance-Covariance Matrix ... ");
                        VarianceCovariance(vc_mat);
                        Console.Write("OK\n");

                        //----- 固有ベクトル, 固有値 -----//
                        Console.Write("Eigenvector, Eigenvalue ... ");
                        Eigen(vc_mat, eval_mat, evect_mat);
                        Console.Write("OK\n");
                    }
                    //----- 固有値の書き出し -----//
                    Console.Write("EigenVal Mat Write ... ");
                    filename = savefolder + "Eval_mat.csv"; // 出力ファイル名(固有値)
                    WriteMat(eval_mat, filename);
                    Console.Write("OK\n");
                }
                //----- 固有顔の書き出し -----//
                Console.Write("Eigen Image Write ... ");
                EigenWrite(evect_mat, ori_evect_mat);
                Console.Write("OK\n");

                //----- 固有ベクトルの書き出し -----//
                Console.Write("EigenVector Mat Write ... ");
                filename = savefolder + "Evect_mat.csv"; // 出力ファイル名(固有ベクトル)
                WriteMat(ori_evect_mat, filename);
                Console.Write("OK\n");
            }
            
            Console.WriteLine(); // 改行
        }

        // 画像サイズ決定
        public void ReadOneImage()
        {
            // 1枚画像を読み込む(サイズの決定)
            filename = folder + "/" + "1.jpg"; //指定した画像形式で読み込む
            src_img = Cv.LoadImage(filename, LoadMode.GrayScale);

            // 行列の設定
            iWidth = src_img.Width;		// 画像の縦幅
            iHeight = src_img.Height;	// 画像の横幅
            numImage = Constants.LEARNING_DATA;
            maxSize = iWidth * iHeight;	// 画像サイズの定義
            src_mat = Cv.CreateMat(maxSize, numImage, MatrixType.F32C1);
            Cv.SetZero(src_mat);		// 初期化
            Console.Write("Image Size [Width:" + iWidth + " Height:" + iHeight + "]\n"); // サイズ確認
            dst_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1);
        }

        // 画像読み込み
        public void ReadImage()
        {
            for (int i = 0; i < numImage; i++)
            {
                // 入力ファイル
                filename = folder + "/" + (i + 1) + ".jpg"; //指定した画像形式で読み込む
                src_img = Cv.LoadImage(filename, LoadMode.GrayScale);

                // 書き出しファイル名
                filename = savefolder + "Lerning/" + String.Format("{0:000}", i + 1) + ".jpg";
                Cv.SaveImage(filename, src_img);

                // 各画素を行列にセット
                for (int y = 0; y < iHeight; y++)
                {
                    for (int x = 0; x < iWidth; x++)
                    {
                        Cv.Set2D(src_mat, iWidth * y + x, i, src_img[y, x]);
                    }
                }
            }
        }

        // 平均行列
        public void Mean()
        {
            // 平均行列の取得
            mean_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1);
            Cv.SetZero(mean_mat);

            // 読み込みデータの平均
            for (int y = 0; y < src_mat.Rows; y++)
                for (int x = 0; x < src_mat.Cols; x++)
                    Cv.Set2D(mean_mat, y, 0, Cv.Get2D(mean_mat, y, 0).Val0 + Cv.Get2D(src_mat, y, x).Val0);

            // スカラー倍
            double dScale = 1 / (double)numImage;
            Cv.ConvertScale(mean_mat, mean_mat, dScale, 0);

            // 行列から画像へ
            MatToImg(mean_mat, dst_img);
            filename = savefolder + "mean_img.jpg";
            Cv.SaveImage(filename, dst_img);
        }

        // 分散(センタリング)
        public void Centering()
        {
            center_mat = Cv.CreateMat(maxSize, numImage, MatrixType.F32C1);
            // センタリング行列
            for (int y = 0; y < center_mat.Rows; y++)
                for (int x = 0; x < center_mat.Cols; x++)
                    Cv.Set2D(center_mat, y, x, (Cv.Get2D(src_mat, y, x).Val0 - Cv.Get2D(mean_mat, y, 0).Val0));
        }

        // 共分散行列
        public void VarianceCovariance(CvMat input_mat)
        {
            // 配列(変数)宣言(センタリング,共分散行列の作成)
            using (CvMat center_matT = Cv.CreateMat(numImage, maxSize, MatrixType.F32C1))
            using (CvMat line_mat = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
            using (CvMat line_matT = Cv.CreateMat(inline, numImage, MatrixType.F32C1))
            using (CvMat vctmp_mat = Cv.CreateMat(numImage, numImage, MatrixType.F32C1))
            {
                Cv.Transpose(center_mat, center_matT);
                Cv.SetZero(input_mat);	// 初期化
                // 中心化(センタリング)
                for (int j = 0; j < maxSize; j++)
                {
                    LineGetMat(center_matT, line_mat, j);
                    Cv.Transpose(line_mat, line_matT);
                    Cv.MatMul(line_mat, line_matT, vctmp_mat);	// 共分散行列 j番目
                    Cv.Add(vctmp_mat, input_mat, input_mat);
                }
            }
            // スカラー倍
            double dScale = 1 / (double)maxSize;
            Cv.ConvertScale(input_mat, input_mat, dScale, 0);
        }

        // 固有ベクトル，固有値
        public void Eigen(CvMat vc_mat, CvMat eval_mat, CvMat evect_mat)
        {
            using (CvMat zero_mat = Cv.CreateMat(numImage, numImage, MatrixType.F32C1))
            {
                // 対称行列の固有ベクトルと固有値を計算する(初期化)
                Cv.SetZero(zero_mat);

                // 固有値問題を解く(cvSVDの方がcvEigenVVより精度が良い)
                Cv.SVD(vc_mat, eval_mat, evect_mat, zero_mat, SvdFlag.ModifyA);
                //Cv.EigenVV(vc_mat, evect_mat, eval_mat, 0.00001);
                //Cv.Transpose(evect_mat, evect_mat); // EigenVVは転置必要
            }
        }

        // 固有顔の書き出し
        public void EigenWrite(CvMat evect_mat, CvMat ori_evect_mat)
        {
            using (CvMat scale_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat evect_matP = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat line_mat = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
            {
                for (int i = 0; i < numImage; i++)
                {
                    // 従来の固有ベクトル値に変換
                    LineGetMat(evect_mat, line_mat, i); // 1列を取り出す
                    Cv.MatMul(center_mat, line_mat, evect_matP);
                    Normalization(evect_matP);  // 正規化
                    ScaleTrans(evect_matP, scale_mat);
                    MatToImg(scale_mat, dst_img);

                    // 固有画像(固有顔)の書き出し
                    filename = savefolder + "Eigen/" + String.Format("{0:000}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dst_img);

                    // 変換した固有ベクトルを格納
                    for (int j = 0; j < maxSize; j++)
                    {
                        Cv.Set2D(ori_evect_mat, j, i, Cv.Get2D(evect_matP, j, 0).Val0);
                    }
                }
            }
        }

        //---関数---//
        // 行列の表示
        public void PrintMat(CvMat input_mat)
        {
            for (int i = 0; i < input_mat.Rows; i++)
            {
                for (int j = 0; j < input_mat.Cols; j++)
                {
                    Console.Write("{0}\t", Cv.Get2D(input_mat, i, j).Val0);
                }
                Console.Write("\n");
            }
        }

        // スケール変換
        public void ScaleTrans(CvMat input_mat, CvMat scale_mat)
        {
            double max = -1, min = 256;
            double val;
            for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            {
                // 最大値,最小値
                val = Cv.Get2D(input_mat, i, 0);
                if (max < val) max = val;
                else if (min > val) min = val;
            }
            for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            {
                // 最大濃度値:255
                val = (((Cv.Get2D(input_mat, i, 0) - min) / (max - min)) * 255);
                if (val > 255) val = 255;
                else if (val < 0) val = 0;
                Cv.Set2D(scale_mat, i, 0, (int)(val + 0.5));
            }
        }

        // 行列から画像へ
        public void MatToImg(CvMat mat, IplImage img)
        {
            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    img[y, x] = mat[img.Width * y + x, 0];
                }
            }
        }

        // 指定した1列(縦1行)取り出す
        public void LineGetMat(CvMat input_mat, CvMat line_mat, int col)
        {
            for (int i = 0; i < input_mat.Rows; i++)
            {   // 1列取り出し
                Cv.Set2D(line_mat, i, 0, Cv.Get2D(input_mat, i, col).Val0);
            }
        }

        // 正規化
        public void Normalization(CvMat input_mat)
        {
            // Eq: [Ui / Sqrt (∑(i->M)ui^2)]
            double normalize;
            double dSum;

            for (int x = 0; x < input_mat.Width; x++)
            {
                dSum = 0;
                for (int y = 0; y < input_mat.Height; y++)
                {
                    dSum = dSum + Cv.Get2D(input_mat, y, x).Val0 * Cv.Get2D(input_mat, y, x).Val0;
                }

                normalize = Math.Sqrt(dSum);

                for (int y = 0; y < input_mat.Height; y++)
                {
                    Cv.Set2D(input_mat, y, x, (Cv.Get2D(input_mat, y, x).Val0) / normalize);
                }
            }
        }

        // 行列の書き出し
        public void WriteMat(CvMat input_mat, string str)
        {
            // ファイルにテキストを書き出し
            using (StreamWriter sw = new StreamWriter(@str))
            {
                // 書き込み
                for (int y = 0; y < input_mat.Height; y++)
                {
                    for (int x = 0; x < input_mat.Width; x++)
                    {
                        sw.Write("{0}, ", Cv.Get2D(input_mat, y, x).Val0);
                    }
                    sw.WriteLine(""); // 改行
                }
            }
        }
    }
}
