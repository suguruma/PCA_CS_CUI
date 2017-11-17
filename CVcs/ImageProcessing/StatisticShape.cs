using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.ImageProcessing
{
    class StatisticShape
    {
        // グローバル宣言
        CvMat src_mat;  // 原画像行列
        CvMat mean_mat, center_mat; // 平均&分散
        string folder, savefolder, filename; // ファイル位置，名前
        int maxSize;
        int dimension;
        int numImage, inline = 1;
        //
        string[] PointFilenames;
        string[] PointFilename;      // パスなしファイル名/拡張子ありファイル名
        string[] _PointFilename;    // 拡張子なしファイル名

        // 主成分分析 (メイン関数)
        public StatisticShape()
        {
            //===== 画像処理(PCA) =====//
            Console.Write("*** Statistic Shape ***\n");
            Console.Write("Please Input Folder Name -> ");

            // ユーザーの入力したフォルダを1行読み込む
            folder = "../../Data/Images/" + "BeautyMaVIC/3/";//Console.ReadLine();
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
            filename = savefolder + "Mean_mat.txt"; // 出力ファイル名(平均ベクトル)
            WritePTS(mean_mat, filename);
            Console.Write("OK\n");

            //----- 分散(センタリング) -----//
            Console.Write("Centering ... ");
            Centering();
            Console.Write("OK\n");

            // src_matと同じ大きさ
            using (CvMat evect_mat = Cv.CreateMat(numImage, numImage, MatrixType.F32C1))
            using (CvMat ori_evect_mat = Cv.CreateMat(dimension, numImage, MatrixType.F32C1))
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
                //----- 固有形状の書き出し -----//
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
            PointFilenames = System.IO.Directory.GetFiles(folder, "*.txt");
            
            // 行列の設定
            numImage = PointFilenames.Count();
            maxSize = 114;	// 画像サイズの定義
            dimension = maxSize * 2;
            src_mat = Cv.CreateMat(dimension, numImage, MatrixType.F32C1);
            Cv.SetZero(src_mat);		    // 初期化
        }

        // 画像読み込み
        public void ReadImage()
        {
            PointFilenames = System.IO.Directory.GetFiles(folder, "*.txt");
            PointFilename = new string[PointFilenames.Count()];     // パスなし
            _PointFilename = new string[PointFilenames.Count()];    // +拡張子なし

            for (int i = 0; i < PointFilenames.Count(); i++)
            {
                string[] SplitText;
                SplitText = PointFilenames[i].Replace("\\", "/").Split('/');
                string StrText = SplitText[SplitText.Count() - 1];
                PointFilename[i] = StrText;
                SplitText = StrText.Split('.');
                _PointFilename[i] = SplitText[0];
                //Console.WriteLine(PointFilename[i]);

                string strText = "";
                using (StreamReader sr = new StreamReader(@PointFilenames[i]))
                {
                    strText = sr.ReadToEnd();
                }
                // string.Splitで分割
                string[] splitText1;
                splitText1 = strText.Replace("\r\n", "\n").Split('\n');

                int read_max_point = maxSize;       // 読み込みの最大点数
                int pts_property = 12;          // プロパティの行数
                int X, Y;
                for (int j = pts_property; j < read_max_point + pts_property; j++)
                {
                    int jj = j - pts_property;
                    string[] splitText2;
                    splitText2 = splitText1[j].Split(' ');
                    X = int.Parse(splitText2[5]);
                    Y = int.Parse(splitText2[6]);

                    // 各画素を行列にセット
                    for (int k = 0; k < 2; k++)
                    {
                        if (k % 2 == 1)
                        {
                            Cv.Set2D(src_mat, jj * 2 + 1, i, Y);
                        }
                        else
                        {
                            Cv.Set2D(src_mat, jj * 2, i, X);
                        }
                    }
                }
            }
            //PrintMat(src_mat);
        }

        // 平均行列
        public void Mean()
        {
            // 平均行列の取得
            mean_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1);
            Cv.SetZero(mean_mat);

            // 読み込みデータの平均
            for (int y = 0; y < src_mat.Rows; y++)
            {
                for (int x = 0; x < src_mat.Cols; x++)
                {
                    Cv.Set2D(mean_mat, y, 0, Cv.Get2D(mean_mat, y, 0).Val0 + Cv.Get2D(src_mat, y, x).Val0);
                }
            }

            // スカラー倍
            double dScale = 1 / (double)numImage;
            Cv.ConvertScale(mean_mat, mean_mat, dScale, 0);
        }

        // 分散(センタリング)
        public void Centering()
        {
            center_mat = Cv.CreateMat(dimension, numImage, MatrixType.F32C1);
            // センタリング行列
            for (int y = 0; y < center_mat.Rows; y++)
            {
                for (int x = 0; x < center_mat.Cols; x++)
                {
                    Cv.Set2D(center_mat, y, x, (Cv.Get2D(src_mat, y, x).Val0 - Cv.Get2D(mean_mat, y, 0).Val0));
                }
            }
        }

        // 共分散行列
        public void VarianceCovariance(CvMat input_mat)
        {
            // 配列(変数)宣言(センタリング,共分散行列の作成)
            using (CvMat center_matT = Cv.CreateMat(numImage, dimension, MatrixType.F32C1))
            using (CvMat line_mat = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
            using (CvMat line_matT = Cv.CreateMat(inline, numImage, MatrixType.F32C1))
            using (CvMat vctmp_mat = Cv.CreateMat(numImage, numImage, MatrixType.F32C1))
            {
                Cv.Transpose(center_mat, center_matT);
                Cv.SetZero(input_mat);	// 初期化
                // 中心化(センタリング)
                for (int j = 0; j < dimension; j++)
                {
                    LineGetMat(center_matT, line_mat, j);
                    Cv.Transpose(line_mat, line_matT);
                    Cv.MatMul(line_mat, line_matT, vctmp_mat);	// 共分散行列 j番目
                    Cv.Add(vctmp_mat, input_mat, input_mat);
                }
            }
            // スカラー倍
            double dScale = 1 / (double)dimension;
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
            using (CvMat scale_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            using (CvMat evect_matP = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            using (CvMat line_mat = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
            {
                for (int i = 0; i < numImage; i++)
                {
                    // 従来の固有ベクトル値に変換
                    LineGetMat(evect_mat, line_mat, i); // 1列を取り出す
                    Cv.MatMul(center_mat, line_mat, evect_matP);
                    Normalization(evect_matP);  // 固有形状顔を表示したいときはコメントアウト

                    // 平均形状に加算
                    Cv.Zero(scale_mat);
                    Cv.Add(scale_mat, mean_mat, scale_mat);
                    Cv.Add(scale_mat, evect_matP, scale_mat);

                    //ScaleTrans(evect_matP, scale_mat);
                    //MatToTxt(scale_mat, dst_img);

                    // 固有画像(固有顔)の書き出し
                    filename = savefolder + "Eigen/" + String.Format("{0:000}", i + 1) + ".txt";
                    WritePTS(scale_mat, filename);

                    // 変換した固有ベクトルを格納
                    for (int j = 0; j < dimension; j++)
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
            double maxX = -100, minX = 3000;
            double maxY = -100, minY = 2000;
            double maxMeanX = -100, minMeanX = 3000;
            double maxMeanY = -100, minMeanY = 2000;
            double val;
            //for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            {
                // 最大値,最小値
                val = Cv.Get2D(input_mat, i, 0);
                if (i % 2 == 0)
                {
                    if (maxX < val) maxX = val;
                    else if (minX > val) minX = val;
                }
                else
                {
                    if (maxY < val) maxY = val;
                    else if (minY > val) minY = val;
                }
            }
            // 平均の最大最小
            for (int i = 0; i < dimension; i++)
            {
                // 最大値,最小値
                val = Cv.Get2D(mean_mat, i, 0);
                if (i % 2 == 0)
                {
                    if (maxMeanX < val) maxMeanX = val;
                    else if (minMeanX > val) minMeanX = val;
                }
                else
                {
                    if (maxMeanY < val) maxMeanY = val;
                    else if (minMeanY > val) minMeanY = val;
                }
            }
            for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            {
                // 最大
                if (i % 2 == 0)
                {
                    val = (((Cv.Get2D(input_mat, i, 0) - minX) / (maxX - minX)) * maxMeanX);
                    if (val > 3000) val = 3000;
                    else if (val < 0) val = 0;
                }
                else
                {
                    val = (((Cv.Get2D(input_mat, i, 0) - minY) / (maxY - minY)) * maxMeanY);
                    if (val > 2000) val = 2000;
                    else if (val < 0) val = 0;
                }
                Cv.Set2D(scale_mat, i, 0, (int)(val + 0.5));
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
        public void WritePTS(CvMat input_mat, string filename)
        {
            // 時間取得
            DateTime dt = DateTime.Now;

            using (StreamWriter sw = new StreamWriter(@filename))
            {
                string str = filename;
                string[] splitText = str.Split('.');

                sw.WriteLine("*********************************************"); // 1行目
                sw.WriteLine("*   FileType : {0} File", splitText[splitText.Count() - 1].ToUpper()); // 4行目
                sw.WriteLine("*   DateTime : {0}.{1}.{2} [{3}:{4}:{5}]", dt.Year, dt.Month, dt.Day, dt.Hour, dt.Minute, dt.Second); // 3行目
                sw.WriteLine("*   Written by AFIMsystem"); // 2行目
                sw.WriteLine("*********************************************"); // 5行目
                sw.WriteLine(""); // 6行目
                sw.WriteLine("# Total Landmark number: {0}", maxSize); // 7行目
                sw.WriteLine(""); // 8行目
                sw.WriteLine("# Total Area number: 10"); // 9行目
                sw.WriteLine("# Area number List: 0 1 2 3 4 5 6 7 8 9"); // 10行目
                sw.WriteLine(""); // 11行目
                sw.WriteLine("# Format: [Area_Number][Index_Numer_in_Area][Index_Numer][X][Y][ConnectFrom][ConnectTo]"); // 12行目

                for (int i = 0; i < maxSize; i++)
                {
                    int X = (int)(Cv.Get2D(input_mat, i * 2, 0) + 0.5);
                    int Y = (int)(Cv.Get2D(input_mat, i * 2 + 1, 0) + 0.5);
                    sw.WriteLine("#  0 0 {0} {1} {2} 0 0", i, X, Y);
                }
            }
        }
    }
}
