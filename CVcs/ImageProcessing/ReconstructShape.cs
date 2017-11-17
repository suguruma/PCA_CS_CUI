using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using KwsmLab.OpenCvSharp;

namespace CVcs.ImageProcessing
{
    class ReconstructShape
    {
        // グローバル宣言
        CvMat mean_mat; // 平均
        CvMat evect_mat;  // 射影行列
        CvMat pjt_mat;  // 射影行列
        CvMat pjt_matT;  // 射影行列
        string folder, savefolder, filename; // ファイル位置，名前
        string inputfile_name;
        int maxSize;
        int dimension;
        int numImage, inline = 1;
        int FlagOfFirst = -1;

        string[] PointFilenames;
        string[] PointFilename;      // パスなしファイル名/拡張子ありファイル名
        string[] _PointFilename;    // 拡張子なしファイル名

        // 再構成画像の名前[string inputstr]

        public ReconstructShape(int StartFile, int RangeOfRec)
        {
            for (int i = StartFile; i < StartFile + RangeOfRec; i++)
            {
                //===== 画像処理(PCA) =====//
                Console.Write("*** Image Processing of PCA ***\n");
                Console.Write("Please Input Folder Name -> ");

                // ユーザーの入力したフォルダを1行読み込む
                folder = "../../Data/Images/" + "BeautyMaVIC/3/";//Console.ReadLine();
                savefolder = "../../Data/Savefile/";

                PointFilenames = System.IO.Directory.GetFiles(folder, "*.txt");
                PointFilename = new string[PointFilenames.Count()];     // パスなし
                _PointFilename = new string[PointFilenames.Count()];    // +拡張子なし
                
                //
                string[] SplitText;
                SplitText = PointFilenames[i].Replace("\\", "/").Split('/');
                string StrText = SplitText[SplitText.Count() - 1];
                PointFilename[i] = StrText;
                SplitText = StrText.Split('.');
                _PointFilename[i] = SplitText[0];

                // フォルダ作成のため
                string inputstr = PointFilename[i];
                inputfile_name = inputstr;


                //----- 画像の読み込み -----//
                Console.Write("Input File ... ");
                ReadOneImage();
                Console.Write("OK\n");

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
                Console.Write("OK\n");

                //----- 投影行列の書き出し -----//
                Console.Write("Projection Mat Write ... ");
                Cv.Transpose(pjt_mat, pjt_matT);
                filename = savefolder + "Confference/" + "cf_(" + inputstr + ")" + ".csv";
                WriteMat(pjt_matT, filename);
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
            PointFilenames = System.IO.Directory.GetFiles(folder, "*.txt");

            // 行列の設定
            numImage = PointFilenames.Count();
            maxSize = 114;	// 画像サイズの定義
            dimension = maxSize * 2;
            mean_mat = Cv.CreateMat(dimension, inline, MatrixType.F32C1);
            Cv.SetZero(mean_mat);		    // 初期化
        }

        // 投影
        public void Projection(CvMat input_mat, string inputstr)
        {
            // 画像から行列へ
            pjt_mat = Cv.CreateMat(inline, numImage, MatrixType.F32C1);
            using (CvMat test_mat = Cv.CreateMat(inline, dimension, MatrixType.F32C1))
            {
                Cv.Zero(test_mat);
                // 射影する画像を入力
                filename = folder + "/" + inputstr; //Console.ReadLine();

                string strText = "";
                using (StreamReader sr = new StreamReader(@filename))
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
                            Cv.Set2D(test_mat, 0, jj * 2 + 1, Y);
                        }
                        else
                        {
                            Cv.Set2D(test_mat, 0, jj * 2, X);
                        }
                    }
                }

                // 入力画像をセンタリングして格納
                for (int i = 0; i < test_mat.Rows * test_mat.Cols; i++)
                {
                    Cv.Set2D(test_mat, 0, i, Cv.Get2D(test_mat, 0, i) - Cv.Get2D(mean_mat, i, 0));
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
                    // 1列を取り出す
                    LineGetMat(input_mat, line_mat, i);
                    Cv.ConvertScale(line_mat, line_mat, Cv.Get2D(pjt_mat, 0, i).Val0, 0);
                    Cv.Add(rec_mat, line_mat, rec_mat);
                    //PrintMat(rec_mat);
                    //ScaleTrans(rec_mat, scale_mat);

                    //filename = savefolder + name_folder + "/" + "test" + ".txt";
                    //WritePTS(rec_mat, filename);

                    // 再構成顔の書き出し(フォルダ付き)
                    name_folder = "Reconstitution/" + "No" + inputfile_name;
                    Directory.CreateDirectory(savefolder + name_folder);
                    filename = savefolder + name_folder + "/" + String.Format("{0:000}", i + 1) + ".txt";
                    WritePTS(rec_mat, filename);

                    // 再構成顔の書き出し(フォルダなし)
                    //filename = savefolder + "Reconstitution/" + inputfile_name;
                    //filename = savefolder + "Reconstitution/" + String.Format("{0:000}", i + 1) + ".jpg";

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

        // 行列の読み込み
        public void ReadMat(CvMat input_mat, string str)
        {
            // ファイルからテキストを読み出し
            using (StreamReader rf = new StreamReader(str))
            {
                string line;
                string[] parts;
                string[] separator = { ", " }; // セパレート文字列(優先順位:左から)
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
