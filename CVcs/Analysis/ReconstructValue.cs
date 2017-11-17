using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.Analysis
{
    class ReconstructValue
    {
        // グローバル宣言
        IplImage src_img, dst_img;  // 画像
        CvMat mean_mat; // 平均
        CvMat evect_mat;  // 射影行列
        CvMat pjt_mat;  // 射影行列
        CvMat pjt_matT;  // 射影行列
        string folder, savefolder, filename; // ファイル位置，名前
        string inputfile_name;
        int iWidth = 0, iHeight = 0, maxSize;
        int dimension;
        int numImage, inline = 1;
        int FlagOfFirst = -1;

        // 再構成画像の名前[string inputstr]

        public ReconstructValue(int StartFile, int RangeOfRec)
        {
            using (CvMat bw_mat = Cv.CreateMat(Constants.LEARNING_DATA, RangeOfRec, MatrixType.F32C1))
            {

                for (int i = StartFile; i < StartFile + RangeOfRec; i++)
                {
                    // フォルダ作成のため
                    string inputstr = i + ".jpg";
                    //string inputstr = i.ToString("000") + ".jpg";
                    inputfile_name = inputstr;

                    //===== 画像処理(PCA) =====//
                    Console.Write("--- Reconstruct RGB ---\n");
                    Console.Write("Please Input Folder Name -> ");

                    // ユーザーの入力したフォルダを1行読み込む
                    //folder = "../../Data/Images/" + Console.ReadLine();

                    Console.Write(inputstr + "\n");
                    folder = "../../Data/Images/" + "MaVICPlus"; // フォルダを指定
                    savefolder = "../../Data/Savefile/";

                    //----- 画像の読み込み -----//
                    ReadOneImage();

                    //----- 平均画像(平均顔) -----//
                    Console.Write("Mean ... ");
                    filename = savefolder + "Mean_mat.csv"; // 入力ファイル名(固有ベクトル)
                    ReadMat(mean_mat, filename);
                    Console.Write("OK\n");

                    if (FlagOfFirst < 0)
                    {
                        // src_matと同じ行列の大きさ
                        evect_mat = Cv.CreateMat(dimension, numImage, MatrixType.F32C1);
                        pjt_matT = Cv.CreateMat(numImage, inline, MatrixType.F32C1);

                        //----- 固有ベクトルの読み込み -----//
                        Console.Write("EigenVector Mat Read ... ");
                        filename = savefolder + "Evect_mat.csv"; // 入力ファイル名(固有ベクトル)
                        ReadMat(evect_mat, filename);
                        Console.Write("OK\n");
                        FlagOfFirst = 1;
                    }

                    //----- 投影(射影) -----//
                    Console.Write("Projection ... ");
                    Projection(evect_mat, inputstr);
                    Cv.Transpose(pjt_mat, pjt_matT);
                    LineSetMat(pjt_matT, bw_mat, i-1);
                    Console.Write("OK\n");

                    //----- 再構成 -----//
                    Console.Write("Reconstitution ... ");
                    //Reconstitution(evect_mat);
                    Reconstitution_Kai(evect_mat, i);
                    Console.Write("OK\n");
                }

                //----- 投影行列の書き出し -----//
                Console.Write("Projection Mat Write ... ");
                filename = savefolder + "Confference.csv";
                WriteMat(bw_mat, filename);
                Console.Write("OK\n");
            }
        }

        // 画像サイズ決定
        public void ReadOneImage()
        {
            // 1枚画像を読み込む(サイズの決定)
            filename = folder + "/" + "1.jpg"; //指定した画像形式で読み込む
            //filename = folder + "/" + "001.jpg"; //指定した画像形式で読み込む
            src_img = Cv.LoadImage(filename, LoadMode.Color);

            // 行列の設定
            iWidth = src_img.Width;		// 画像の縦幅
            iHeight = src_img.Height;	// 画像の横幅
            numImage = Constants.LEARNING_DATA;
            maxSize = iWidth * iHeight;	// 画像サイズの定義
            dimension = maxSize * 3;
            Console.Write("Image Size [Width:" + iWidth + " Height:" + iHeight + "]\n"); // サイズ確認
            dst_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 3);
            
            // 平均行列の取得
            mean_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1);
        }

        // 投影
        public void Projection(CvMat input_mat, string inputstr)
        {
            // 画像から行列へ
            pjt_mat = Cv.CreateMat(inline, numImage, MatrixType.F32C1);
            using (CvMat test_mat = Cv.CreateMat(inline, dimension, MatrixType.F32C1))
            {
                // 射影する画像を入力
                filename = folder + "/" + inputstr; //Console.ReadLine();
                using (IplImage input_img = Cv.LoadImage(filename, LoadMode.Color))
                {
                    // 各画素を行列にセット(横1行で取得する)
                    for (int j = 0; j < 3; j++)
                    {
                        for (int y = 0; y < iHeight; y++)
                        {
                            for (int x = 0; x < iWidth; x++)
                            {
                                if ((iWidth * y + x + j * maxSize) % 3 == 0)
                                {   // R:0,3,6....
                                    Cv.Set2D(test_mat, 0,
                                        iWidth * y + x + j * maxSize, input_img[y, x].Val2); // R
                                }
                                else if ((iWidth * y + x + j * maxSize) % 3 == 1)
                                {   // G:1,4,7....
                                    Cv.Set2D(test_mat, 0,
                                        iWidth * y + x + j * maxSize, input_img[y, x].Val1); // G
                                }
                                else
                                {   // B:2,5,8....
                                    Cv.Set2D(test_mat, 0,
                                        iWidth * y + x + j * maxSize, input_img[y, x].Val0); // B
                                }
                            }
                        }
                    }

                    // 入力画像をセンタリングして格納
                    for (int i = 0; i < test_mat.Rows * test_mat.Cols; i++)
                    {
                        Cv.Set2D(test_mat, 0, i, Cv.Get2D(test_mat, 0, i) - Cv.Get2D(mean_mat, i, 0));
                    }
                }
                Cv.MatMul(test_mat, input_mat, pjt_mat);
            }
        }

        // 再構成
        public void Reconstitution(CvMat input_mat)
        {
            string name_folder = "";
            using (CvMat line_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            using (CvMat scale_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            using (CvMat rec_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            {
                Cv.Zero(rec_mat); // 初期化(0)
                Cv.Add(rec_mat, mean_mat, rec_mat); // 平均顔を足す
                for (int i = 0; i < Constants.RECON_DATA; i++)
                {
                    // 指定した基底を抜き出して再構成
                    //if (i == 1 && i == 2) i++;

                    // 1列を取り出す
                    LineGetMat(input_mat, line_mat, i);
                    Cv.ConvertScale(line_mat, line_mat, Cv.Get2D(pjt_mat, 0, i).Val0, 0);
                    Cv.Add(rec_mat, line_mat, rec_mat);
                    ScaleTrans_Kai(rec_mat, scale_mat);
                    MatToImg(scale_mat, dst_img);

                    // 再構成顔の書き出し(フォルダ付き)
                    name_folder = "Reconstitution/" + "No" + inputfile_name;
                    Directory.CreateDirectory(savefolder + name_folder);
                    filename = savefolder + name_folder + "/" + String.Format("{0:000}", i + 1) + ".jpg";

                    // 再構成顔の書き出し(フォルダなし)
                    //filename = savefolder + "Reconstitution/" + inputfile_name;
                    
                    //filename = savefolder + "Reconstitution/" + String.Format("{0:000}", i + 1) + ".jpg";
                        
                    Cv.SaveImage(filename, dst_img);
                }
            }
        }

        // 再構成改
        public void Reconstitution_Kai(CvMat input_mat, int no_img)
        {
            string name_folder = "";
            using (CvMat line_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            using (CvMat scale_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            using (CvMat rec_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1))
            {
                Cv.Zero(rec_mat); // 初期化(0)
                Cv.Add(rec_mat, mean_mat, rec_mat); // 平均顔を足す
                for (int i = 0; i < Constants.RECON_DATA; i++)
                {
                    //Cv.Zero(rec_mat); // 初期化(0)
                    
                    // 1列を取り出す
                    LineGetMat(input_mat, line_mat, i);
                    Cv.ConvertScale(line_mat, line_mat, Cv.Get2D(pjt_mat, 0, i).Val0, 0);
                    Cv.Add(rec_mat, line_mat, rec_mat);

                    //ScaleTrans(rec_mat, scale_mat);
                    //MatToImg(scale_mat, dst_img);

                    // 再構成顔の書き出し(フォルダ付き)
                    //name_folder = "Reconstitution/" + inputfile_name;
                    //Directory.CreateDirectory(savefolder + name_folder);
                    //filename = savefolder + name_folder + "/" + String.Format("{0:000}", i + 1) + ".jpg";

                    // 再構成顔の書き出し(フォルダなし)
                    //filename = savefolder + "Reconstitution/" + inputfile_name;

                    if (i == 2) //(累積寄与率70%)
                    {
                        ScaleTrans(rec_mat, scale_mat);
                        MatToImg(scale_mat, dst_img);
                        filename = savefolder + "Reconstitution/" + String.Format("{0:000}", no_img) + ".jpg";
                        Cv.SaveImage(filename, dst_img);
                        break;
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
                val = (((Cv.Get2D(input_mat, i, 0) - min) / (max - min)) * 255); // 平均顔を加えない場合
                //val = (((Cv.Get2D(input_mat, i, 0) - min) / (max - min)) * max);
                if (val > 255) val = 255;
                else if (val < 0) val = 0;
                Cv.Set2D(scale_mat, i, 0, (int)(val + 0.5));
            }
        }
        
        // スケール変換改
        public void ScaleTrans_Kai(CvMat input_mat, CvMat scale_mat)
        {
            double maxR = -1, minR = 256;
            double maxG = -1, minG = 256;
            double maxB = -1, minB = 256;
            double val;
            for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            {
                // 最大値,最小値
                val = Cv.Get2D(input_mat, i, 0);
                if (i % 3 == 0)
                {
                    if (maxR < val) maxR = val;
                    else if (minR > val) minR = val;
                }
                else if (i % 3 == 1)
                {
                    if (maxG < val) maxG = val;
                    else if (minG > val) minG = val;
                }
                else
                {
                    if (maxB < val) maxB = val;
                    else if (minB > val) minB = val;
                }
            }
            for (int i = 0; i < input_mat.Rows * input_mat.Cols; i++)
            {
                // 最大濃度値:255
                if (i % 3 == 0)
                    val = (((Cv.Get2D(input_mat, i, 0) - minR) / (maxR - minR)) * maxR);
                else if (i % 3 == 1)
                    val = (((Cv.Get2D(input_mat, i, 0) - minG) / (maxG - minG)) * maxG);
                else
                    val = (((Cv.Get2D(input_mat, i, 0) - minB) / (maxB - minB)) * maxB);
                if (val > 255) val = 255;
                else if (val < 0) val = 0;
                Cv.Set2D(scale_mat, i, 0, (int)(val + 0.5));
            }
            /*
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
                val = (((Cv.Get2D(input_mat, i, 0) - min) / (max - min)) * 255); // 平均顔を加えない場合
                //val = (((Cv.Get2D(input_mat, i, 0) - min) / (max - min)) * max);
                if (val > 255) val = 255;
                else if (val < 0) val = 0;
                Cv.Set2D(scale_mat, i, 0, (int)(val + 0.5));
            }*/
        }

        // 行列から画像へ
        public void MatToImg(CvMat input_mat, IplImage img)
        {
            using (IplImage imgR = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1))
            using (IplImage imgG = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1))
            using (IplImage imgB = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1))
            {
                // 各画素を行列にセット
                for (int i = 0; i < 3; i++)
                {
                    for (int y = 0; y < img.Height; y++)
                    {
                        for (int x = 0; x < img.Width; x++)
                        {
                            if ((iWidth * y + x + i * maxSize) % 3 == 0)
                            {   // R:0,3,6....
                                imgR[y, x] = input_mat[img.Width * y + x + i * maxSize, 0]; // R
                            }
                            else if ((iWidth * y + x + i * maxSize) % 3 == 1)
                            {   // G:1,4,7....
                                imgG[y, x] = input_mat[img.Width * y + x + i * maxSize, 0]; // G
                            }
                            else
                            {   // B:2,5,8....
                                imgB[y, x] = input_mat[img.Width * y + x + i * maxSize, 0]; // B
                            }
                        }
                    }
                }
                Cv.Merge(imgB, imgG, imgR, null, img);
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

        // 指定した1列(縦1行)セットする
        public void LineSetMat(CvMat line_mat, CvMat output_mat, int col)
        {
            for (int i = 0; i < line_mat.Rows; i++)
            {   // 1列取り出し
                Cv.Set2D(output_mat, i, col, Cv.Get2D(line_mat, i, 0).Val0);
            }
        }

        // 行列の読み込み
        public void ReadMat(CvMat input_mat, string str)
        {
            // ファイルからテキストを読み出し
            using (StreamReader rf = new StreamReader(str))
            {
                string line;
                string[] parts;
                string[] separator = {", "}; // セパレート文字列(優先順位:左から)
                int x, y;

                y = 0;
                while ((line = rf.ReadLine()) != null) // 1行ずつ読み出し。
                {
                    x = 0;
                    parts = line.Split(separator, StringSplitOptions.RemoveEmptyEntries);
                    foreach (string s in parts)
                    {
                        // 読み込んだsをdouble型に変換
                        Cv.Set2D(input_mat, y, x, double.Parse(s));
                        x++;
                    }
                    y++;
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
                        if (x != 0 && x != input_mat.Width) sw.Write(",");
                        sw.Write("{0} ", Cv.Get2D(input_mat, y, x).Val0);
                    }
                    sw.WriteLine(""); // 改行
                }
            }
        }
    }
}
