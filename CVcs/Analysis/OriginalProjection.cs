﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.Analysis
{
    class OriginalProjection
    {
        // グローバル宣言
        IplImage src_img, dst_img;  // 画像
        CvMat mean_mat; // 平均
        CvMat pjt_mat;  // 射影行列
        string folder, savefolder, filename; // ファイル位置，名前
        int iWidth = 0, iHeight = 0, maxSize;
        int numImage, inline = 1;
        // 再構成画像の名前[string inputstr]

        public OriginalProjection(string inputstr)
        {
            //===== 画像処理(PCA) =====//
            Console.Write("--- OriginalProjection ---\n");
            Console.Write("Please Input Folder Name -> ");

            // ユーザーの入力したフォルダを1行読み込む
            folder = "../../Data/Images/" + Console.ReadLine();

            Console.Write(inputstr + "\n");
            //folder = "../../Data/Images/" + "ImageB"; // フォルダを指定
            savefolder = "../../Data/Savefile/";

            //----- 画像の読み込み -----//
            ReadOneImage();

            //----- 平均画像(平均顔) -----//
            Console.Write("Mean ... ");
            filename = savefolder + "Mean_mat.csv"; // 入力ファイル名(平均ベクトル)
            ReadMat(mean_mat, filename);
            Console.Write("OK\n");

            // src_matと同じ大きさ
            using (CvMat evect_mat = Cv.CreateMat(maxSize, numImage, MatrixType.F32C1))
            using (CvMat pjt_matT = Cv.CreateMat(numImage, inline, MatrixType.F32C1))
            {
                //----- 固有ベクトルの読み込み -----//
                Console.Write("EigenVector Mat Read ... ");
                filename = savefolder + "Evect_mat.csv"; // 入力ファイル名(固有ベクトル)
                ReadMat(evect_mat, filename);
                Console.Write("OK\n");

                //----- 投影(射影) -----//
                Console.Write("Projection ... ");
                Projection(evect_mat, inputstr);
                Console.Write("OK\n");

                //----- 投影行列の書き出し -----//
                Console.Write("Projection Mat Write ... ");
                Cv.Transpose(pjt_mat, pjt_matT);
                filename = savefolder + "Confference/" + "cf_(" + inputstr + ")" + ".csv";
                WriteMat(pjt_matT, filename);
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
            Console.Write("Image Size [Width:" + iWidth + " Height:" + iHeight + "]\n"); // サイズ確認
            dst_img = Cv.CreateImage(Cv.Size(iWidth, iHeight), BitDepth.U8, 1);

            // 平均行列の取得
            mean_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1);
        }

        // 投影
        public void Projection(CvMat input_mat, string inputstr)
        {
            // 画像から行列へ
            pjt_mat = Cv.CreateMat(inline, numImage, MatrixType.F32C1);
            using (CvMat test_mat = Cv.CreateMat(inline, maxSize, MatrixType.F32C1))
            {
                // 射影する画像を入力
                filename = folder + "/" + inputstr; //Console.ReadLine();
                using (IplImage input_img = Cv.LoadImage(filename, LoadMode.GrayScale))
                {
                    // 入力画像をセンタリングして格納(横1行で取得する)
                    for (int y = 0; y < input_img.Height; y++)
                        for (int x = 0; x < input_img.Width; x++)
                            Cv.Set2D(test_mat, 0, x + y * input_img.Width,
                                input_img[y, x] - Cv.Get2D(mean_mat, x + y * input_img.Width, 0));
                }
                Cv.MatMul(test_mat, input_mat, pjt_mat);
            }
        }

        // 再構成
        public void Reconstitution(CvMat input_mat)
        {
            using (CvMat line_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat scale_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            using (CvMat rec_mat = Cv.CreateMat(maxSize, inline, MatrixType.F32C1))
            {
                Cv.Zero(rec_mat); // 初期化(0)
                Cv.Add(rec_mat, mean_mat, rec_mat); // 平均顔を足す
                for (int i = 0; i < numImage; i++)
                {
                    // 指定した基底を抜き出して再構成
                    //if (i >= 0 && i < 10) i++;

                    // 1列を取り出す
                    LineGetMat(input_mat, line_mat, i);
                    Cv.ConvertScale(line_mat, line_mat, Cv.Get2D(pjt_mat, 0, i).Val0, 0);
                    Cv.Add(rec_mat, line_mat, rec_mat);
                    ScaleTrans(rec_mat, scale_mat);
                    MatToImg(scale_mat, dst_img);

                    // 再構成顔の書き出し
                    filename = savefolder + "Reconstitution/" + String.Format("{0:000}", i + 1) + ".jpg";
                    Cv.SaveImage(filename, dst_img);
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
                        sw.Write("{0} ", Cv.Get2D(input_mat, y, x).Val0);
                    }
                    sw.WriteLine(""); // 改行
                }
            }
        }
    }
}
