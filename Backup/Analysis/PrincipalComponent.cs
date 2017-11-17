using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.Analysis
{
    class PrincipalComponent
    {
        // グローバル宣言
        IplImage src_img, dst_img;  // 画像
        CvMat src_mat, mean_mat;    // 行列
        string folder, savefolder, filename; // ファイル位置，名前
        string inputstr = Constants.RECON_NAME;  // 再構成画像の名前
        int iWidth = 0, iHeight = 0, numImage;
        int maxSize;
        int inline = 1;
        double[] confference = new double[Constants.RECON_DATA];

        // 主成分分析 (メイン関数)
        public PrincipalComponent()
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
            Console.Write("Mean Data ... ");
            MeanImage();
            Console.Write("OK\n");

            using (CvMat evect_mat = Cv.CreateMat(maxSize, maxSize, MatrixType.F32C1))
            {
                using (CvMat eval_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
                {
                    using (CvMat vc_mat = Cv.CreateMat(maxSize, maxSize, MatrixType.F32C1))
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
                    //----- 固有ベクトルと固有値の書き出し -----//
                    Console.Write("Eigen Mat Write ... ");
                    EigenText(eval_mat, evect_mat);
                    Console.Write("OK\n");
                }
                //----- 固有顔の書き出し -----//
                Console.Write("Eigen Image Write ... ");
                EigenWrite(evect_mat);
                Console.Write("OK\n");

                //----- 投影(射影) -----//
                Console.Write("Projection ... ");
                Projection(evect_mat);
                Console.Write("OK\n");

                //----- 再構成 -----//
                Console.Write("Reconstitution ... ");
                Reconstitution(evect_mat);
                Console.Write("OK\n");
            }
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
            for (int i = 0; i < Constants.LEARNING_DATA; i++)
            {
                // 入力ファイル
                filename = folder + "/" + (i + 1) + ".jpg"; //指定した画像形式で読み込む
                src_img = Cv.LoadImage(filename, LoadMode.GrayScale);

                // 書き出しファイル名
                filename = savefolder + "Lerning/" + String.Format("{0:000}", i + 1) + ".jpg";
                Cv.SaveImage(filename, src_img);

                // 各画素を行列にセット
                for (int y = 0; y < iHeight; y++)
                    for (int x = 0; x < iWidth; x++)
                        Cv.Set2D(src_mat, iWidth * y + x, i, src_img[y, x]);
            }
        }

        // 平均行列と画像の取得
        public void MeanImage()
        {
            // 平均行列の取得
            mean_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1);	// 行列の１要素が32ビット浮動小数点型
            Cv.SetZero(mean_mat);
            for (int y = 0; y < src_mat.Rows; y++)
            {
                for (int x = 0; x < src_mat.Cols; x++)
                {
                    // 読み込みデータの平均
                    Cv.Set2D(mean_mat, y, 0, Cv.Get2D(mean_mat, y, 0).Val0 + Cv.Get2D(src_mat, y, x).Val0);
                }
            }
            // スカラー倍
            double dScale = 1 / (double)numImage;
            Cv.ConvertScale(mean_mat, mean_mat, dScale, 0);

            // 行列から画像へ
            MatToImg(mean_mat, dst_img);
            filename = savefolder + "mean_img.jpg";
            Cv.SaveImage(filename, dst_img);
        }

        // 共分散行列
        public void VarianceCovariance(CvMat vc_mat)
        {
            // 配列(変数)宣言(センタリング,共分散行列の作成)
            using (CvMat vctmp_mat = Cv.CreateMat(maxSize, maxSize, MatrixType.F32C1))
            using (CvMat center_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat center_matT = Cv.CreateMat(inline, maxSize, MatrixType.F32C1))
            {
                Cv.SetZero(vc_mat);	// 初期化
                // 中心化(センタリング)
                for (int j = 0; j < numImage; j++)
                {
                    Centering(src_mat, center_mat, j);
                    Cv.Transpose(center_mat, center_matT);		// センタリング行列の転置
                    Cv.MatMul(center_mat, center_matT, vctmp_mat);	// 共分散行列 j番目
                    Cv.Add(vctmp_mat, vc_mat, vc_mat);
                }
            }
            // スカラー倍
            double dScale = 1 / (double)numImage;
            Cv.ConvertScale(vc_mat, vc_mat, dScale, 0);
        }

        // 固有ベクトル，固有値
        public void Eigen(CvMat vc_mat, CvMat eval_mat, CvMat evect_mat)
        {
            using (CvMat zero_mat = Cv.CreateMat(maxSize, maxSize, MatrixType.F32C1))
            {
                // 対称行列の固有ベクトルと固有値を計算する(初期化)
                Cv.SetZero(eval_mat);
                Cv.SetZero(evect_mat);
                Cv.SetZero(zero_mat);

                // 固有値問題を解く(cvSVDの方がcvEigenVVより精度が良い)
                Cv.SVD(vc_mat, eval_mat, evect_mat, zero_mat, SvdFlag.ModifyA);
                //Cv.EigenVV(vc_mat, evect_mat, eval_mat, 0.00001);
                //Cv.Transpose(evect_mat, evect_mat); // EigenVVは転置必要
            }
        }

        // 固有ベクトル，固有値の書き出し
        public void EigenText(CvMat eval_mat, CvMat evect_mat)
        {
            // 出力ファイル名(固有ベクトル)
            filename = savefolder + "Evect_mat.txt";

            // ファイルにテキストを書き出し。
            using (StreamWriter sw = new StreamWriter(@filename))
            {
                // 書き込み
                for (int y = 0; y < evect_mat.Height; y++)
                {
                    for (int x = 0; x < evect_mat.Width; x++)
                    {
                        sw.Write("{0} ", Cv.Get2D(evect_mat, y, x).Val0);
                    }
                    sw.WriteLine(""); // 改行
                }
            }

            // 出力ファイル名(固有値)
            filename = savefolder + "Eval_mat.txt";

            // 書き込み
            using (StreamWriter sw = new StreamWriter(@filename))
            {
                for (int y = 0; y < eval_mat.Height; y++)
                {
                    for (int x = 0; x < eval_mat.Width; x++)
                    {
                        sw.Write("{0} ", Cv.Get2D(eval_mat, y, x).Val0);
                    }
                    sw.WriteLine(""); // 改行
                }
            }
        }

        // 固有顔の書き出し
        public void EigenWrite(CvMat evect_mat)
        {
            using (CvMat line_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat scale_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            {
                for (int i = 0; i < Constants.EIGEN_DATA; i++)
                {
                    // 1列を取り出す
                    LineGetMat(evect_mat, line_mat, i);
                    ScaleTrans(line_mat, scale_mat);
                    MatToImg(scale_mat, dst_img);

                    // 固有画像(固有顔)の書き出し
                    filename = savefolder + "Eigen/" + String.Format("{0:000}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dst_img);
                }
            }
        }

        // 投影
        public void Projection(CvMat evect_mat)
        {
            // 画像から行列へ
            using (CvMat input_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat center_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat center_matT = Cv.CreateMat(inline, maxSize, MatrixType.F32C1))
            {
                // 射影する画像を入力
                filename = folder + "/" + inputstr; //Console.ReadLine();
                using (IplImage input_img = Cv.LoadImage(filename, LoadMode.GrayScale))
                {
                    for (int y = 0; y < input_img.Height; y++)
                    {
                        for (int x = 0; x < input_img.Width; x++)
                        {
                            Cv.Set2D(input_mat, x + y * input_img.Width, 0, input_img[y, x]);
                        }
                    }
                }
                // センタリング
                Centering(input_mat, center_mat, 0);
                Cv.Transpose(center_mat, center_matT); // 横1行で取得する

                using (CvMat line_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
                using (CvMat pjt_mat = Cv.CreateMat(inline, inline, MatrixType.F32C1))
                {
                    filename = savefolder + "Confference_mat.txt";

                    using (StreamWriter sw = new StreamWriter(@filename))
                    {
                        for (int i = 0; i < Constants.RECON_DATA; i++)
                        {
                            // 1列を取り出す
                            LineGetMat(evect_mat, line_mat, i);
                            Cv.MatMul(center_matT, line_mat, pjt_mat);
                            confference[i] = Cv.Get2D(pjt_mat, 0, 0).Val0;
                            sw.WriteLine("{0} ", Cv.Get2D(pjt_mat, 0, 0).Val0);
                        }
                    } // using
                } // using
            } // using
        }

        // 再構成
        public void Reconstitution(CvMat evect_mat)
        {
            using (CvMat line_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat scale_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat rec_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            {
                Cv.Zero(rec_mat); // 初期化(0)
                Cv.Add(rec_mat, mean_mat, rec_mat); // 平均顔を足す
                for (int i = 0; i < Constants.RECON_DATA; i++)
                {
                    // 1列を取り出す
                    LineGetMat(evect_mat, line_mat, i);
                    Cv.ConvertScale(line_mat, line_mat, confference[i], 0);
                    Cv.Add(rec_mat, line_mat, rec_mat);
                    ScaleTrans(rec_mat, scale_mat);
                    //MatToImg(scale_mat, dst_img);
                    MatToImg(rec_mat, dst_img);

                    // 再構成顔の書き出し
                    filename = savefolder + "Reconstitution/" + String.Format("{0:000}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dst_img);
                }
            }
        }

        //---関数---//
        // 行列の表示
        public void printMat(CvMat data)
        {
            for (int i = 0; i < data.Rows; i++)
            {
                for (int j = 0; j < data.Cols; j++)
                {
                    Console.Write("{0}\t", Cv.Get2D(data, i, j).Val0);
                }
                Console.Write("\n");
            }
        }

        // スケール変換
        public void ScaleTrans(CvMat data, CvMat scale_mat)
        {
            double max = -1, min = 256;
            double val;
            for (int i = 0; i < data.Rows * data.Cols; i++)
            {
                // 最大値,最小値
                val = Cv.Get2D(data, i, 0);
                if (max < val) max = val;
                else if (min > val) min = val;
            }
            for (int i = 0; i < data.Rows * data.Cols; i++)
            {
                // 最大濃度値:255
                val = (((Cv.Get2D(data, i, 0) - min) / (max - min)) * 255);
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

        // センタリング
        public void Centering(CvMat data, CvMat center_mat, int col)
        {
            for (int i = 0; i < maxSize; i++)
                Cv.Set2D(center_mat, i, 0,
                    (Cv.Get2D(data, i, col) - Cv.Get2D(mean_mat, i, 0)));
        }

        // 指定した1列(縦1行)取り出す
        public void LineGetMat(CvMat data, CvMat line_mat, int col)
        {
            for (int i = 0; i < data.Rows; i++)
            {   // 1列取り出し
                Cv.Set2D(line_mat, i, 0, Cv.Get2D(data, i, col).Val0);
            }
        }
    }
}
