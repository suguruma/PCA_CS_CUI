using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.ImageProcessing
{
    class Aligenment
    {
        int SAMPLENUM = 2;      // 読み込み枚数
        int DIMENTION = 2;      // 次元数(x,y座標)
        int POINTNUM = 103;     // 点の数

        public Aligenment()
        {

            CvMat[] A_mat = new CvMat[SAMPLENUM];
            CvMat[] Z_mat = new CvMat[SAMPLENUM];

            for (int i = 0; i < SAMPLENUM; i++)
            {
                A_mat[i] = Cv.CreateMat(DIMENTION * POINTNUM, 4, MatrixType.F32C1);
                Z_mat[i] = Cv.CreateMat(4, 1, MatrixType.F32C1);
            }

            //--- 基準座標データの読み込み ---//

            // 基準画像の座標を行列Aに格納
            // 基準画像の座標を行列Xに格納
            CvMat X1_mat = Cv.CreateMat(DIMENTION * POINTNUM, 1, MatrixType.F32C1);
            CvMat X2_mat = Cv.CreateMat(DIMENTION * POINTNUM, 1, MatrixType.F32C1);

            String InputPTS = "Data/1.txt";
            string strText = "";
            using (StreamReader sr = new StreamReader(@InputPTS))
            {
                strText = sr.ReadToEnd();
            }
            // string.Splitで分割
            string[] splitText1;
            splitText1 = strText.Replace("\r\n", "\n").Split('\n');

            int read_max_point = POINTNUM;       // 読み込みの最大点数
            int pts_property = 12;          // プロパティの行数
            int X, Y;
            int point = 0;
            for (int i = pts_property; i < read_max_point + pts_property; i++)
            {
                //Console.WriteLine(splitText1[i]);
                string[] splitText2;
                splitText2 = splitText1[i].Split(' ');
                X = int.Parse(splitText2[5]);
                Y = int.Parse(splitText2[6]);

                // 行列Aを格納
                Cv.Set2D(A_mat[0], point * 2, 0, X);
                Cv.Set2D(A_mat[0], point * 2, 1, -Y);
                Cv.Set2D(A_mat[0], point * 2, 2, 1);
                Cv.Set2D(A_mat[0], point * 2, 3, 0);
                Cv.Set2D(A_mat[0], point * 2 + 1, 0, Y);
                Cv.Set2D(A_mat[0], point * 2 + 1, 1, X);
                Cv.Set2D(A_mat[0], point * 2 + 1, 2, 0);
                Cv.Set2D(A_mat[0], point * 2 + 1, 3, 1);

                // 行列Xに座標を格納
                Cv.Set2D(X1_mat, point * 2, 0, X);
                Cv.Set2D(X1_mat, point * 2 + 1, 0, Y);

                point++;
            }

            //--- 変換座標データの読み込み ---//
            String OutputPTS = "Data/2.txt";
            strText = "";
            using (StreamReader sr = new StreamReader(@OutputPTS))
            {
                strText = sr.ReadToEnd();
            }
            // string.Splitで分割
            splitText1 = strText.Replace("\r\n", "\n").Split('\n');

            point = 0;
            for (int i = pts_property; i < read_max_point + pts_property; i++)
            {
                //Console.WriteLine(splitText1[i]);
                string[] splitText2;
                splitText2 = splitText1[i].Split(' ');
                X = int.Parse(splitText2[5]);
                Y = int.Parse(splitText2[6]);

                // 行列Aを格納
                Cv.Set2D(A_mat[1], point * 2, 0, X);
                Cv.Set2D(A_mat[1], point * 2, 1, -Y);
                Cv.Set2D(A_mat[1], point * 2, 2, 1);
                Cv.Set2D(A_mat[1], point * 2, 3, 0);
                Cv.Set2D(A_mat[1], point * 2 + 1, 0, Y);
                Cv.Set2D(A_mat[1], point * 2 + 1, 1, X);
                Cv.Set2D(A_mat[1], point * 2 + 1, 2, 0);
                Cv.Set2D(A_mat[1], point * 2 + 1, 3, 1);


                point++;
            }

            //--- 行列W(重み)を求める(全て1) ---//
            CvMat W_mat = Cv.CreateMat(DIMENTION * POINTNUM, DIMENTION * POINTNUM, MatrixType.F32C1);
            for (int i = 0; i < POINTNUM * DIMENTION; i++)
            {
                for (int j = 0; j < POINTNUM * DIMENTION; j++)
                {
                    if (i == j)
                        Cv.Set2D(W_mat, i, j, 1);
                    else
                        Cv.Set2D(W_mat, i, j, 0);
                }
            }

            //--- アライメント行列計算 ---//
            CvMat tmp1 = Cv.CreateMat(4, DIMENTION * POINTNUM, MatrixType.F32C1);
            CvMat tmp1_t = Cv.CreateMat(DIMENTION * POINTNUM, 4, MatrixType.F32C1);
            CvMat tmp2 = Cv.CreateMat(4, 4, MatrixType.F32C1);

            // 転置行列の準備
            CvMat W_mat_t = Cv.CreateMat(DIMENTION * POINTNUM, DIMENTION * POINTNUM, MatrixType.F32C1);
            Cv.Transpose(W_mat, W_mat_t);

            CvMat A_mat_t = Cv.CreateMat(4, DIMENTION * POINTNUM, MatrixType.F32C1);
            Cv.Transpose(A_mat[1], A_mat_t);


            for (int i = 1; i < SAMPLENUM; i++)
            {
                Cv.MatMul(A_mat_t, W_mat_t, tmp1);
                Cv.MatMul(tmp1, W_mat, tmp1);
                Cv.MatMul(tmp1, A_mat[i], tmp2);
                Cv.Invert(tmp2, tmp2);
                Cv.MatMul(tmp2, A_mat_t, tmp1);
                Cv.MatMul(tmp1, W_mat_t, tmp1);
                Cv.MatMul(tmp1, W_mat, tmp1);
                Cv.MatMul(tmp1, X1_mat, Z_mat[i]);
            }
            Cv.MatMul(A_mat[1], Z_mat[1], X2_mat);

            //--- アライメント計算---//
            CvMat M_mat = Cv.CreateMat(2, 2, MatrixType.F32C1);
            CvMat T_mat = Cv.CreateMat(2, 1, MatrixType.F32C1);

            // 行列Mと行列Tにパラメータを格納
            Cv.Set2D(M_mat, 0, 0, Cv.Get2D(Z_mat[1], 0, 0));
            Cv.Set2D(M_mat, 0, 1, -Cv.Get2D(Z_mat[1], 1, 0));
            Cv.Set2D(M_mat, 1, 0, Cv.Get2D(Z_mat[1], 1, 0));
            Cv.Set2D(M_mat, 1, 1, Cv.Get2D(Z_mat[1], 0, 0));
            Cv.Set2D(T_mat, 0, 0, Cv.Get2D(Z_mat[1], 2, 0));
            Cv.Set2D(T_mat, 1, 0, Cv.Get2D(Z_mat[1], 3, 0));

            // 入力ファイル
            string filename = "Data/2.bmp";
            IplImage src_img = Cv.LoadImage(filename, LoadMode.Color);
            IplImage dst_img = Cv.CloneImage(src_img);
            Cv.Zero(dst_img);

            // 行列の設定
            int iWidth = src_img.Width;		// 画像の縦幅
            int iHeight = src_img.Height;	// 画像の横幅

            CvMat M_mat_inv = Cv.CreateMat(2, 2, MatrixType.F32C1);
            Cv.Invert(M_mat, M_mat_inv);

            CvMat Image_mat = Cv.CreateMat(2, 1, MatrixType.F32C1);
            for (int i = 0; i < iHeight; i++)
            {
                for (int j = 0; j < iWidth; j++)
                {
                    Cv.Set2D(Image_mat, 0, 0, j);
                    Cv.Set2D(Image_mat, 1, 0, i);

                    Cv.Sub(Image_mat, T_mat, Image_mat);
                    Cv.MatMul(M_mat_inv, Image_mat, Image_mat);
                    //PrintMat(Image_mat);

                    if (0 <= Cv.Get2D(Image_mat, 0, 0) && Cv.Get2D(Image_mat, 0, 0) <= iWidth)
                    {
                        if (0 <= Cv.Get2D(Image_mat, 1, 0) && Cv.Get2D(Image_mat, 1, 0) <= iHeight)
                        {
                            dst_img[i, j] = src_img[(int)Cv.Get2D(Image_mat, 1, 0), (int)Cv.Get2D(Image_mat, 0, 0)];
                        }
                    }
                }
            }

            Cv.SaveImage("Data/out.jpg", dst_img);
            WritePTS(X2_mat);
            // 終了
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

        public void WritePTS(CvMat input_mat)
        {
            // 時間取得
            DateTime dt = DateTime.Now;

            string filename = "Data/out.txt";
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
                sw.WriteLine("# Total Landmark number: {0}", POINTNUM); // 7行目
                sw.WriteLine(""); // 8行目
                sw.WriteLine("# Total Area number: 10"); // 9行目
                sw.WriteLine("# Area number List: 0 1 2 3 4 5 6 7 8 9"); // 10行目
                sw.WriteLine(""); // 11行目
                sw.WriteLine("# Format: [Area_Number][Index_Numer_in_Area][Index_Numer][X][Y][ConnectFrom][ConnectTo]"); // 12行目

                for (int i = 0; i < POINTNUM; i++)
                {
                    int X = (int)(Cv.Get2D(input_mat, i * DIMENTION, 0) + 0.5);
                    int Y = (int)(Cv.Get2D(input_mat, i * DIMENTION + 1, 0) + 0.5);
                    sw.WriteLine("#  0 0 {0} {1} {2} 0 0", i, X, Y);
                }
            }
        }
    }
}
