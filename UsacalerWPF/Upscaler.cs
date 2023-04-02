using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Emgu.CV;
using Emgu.CV.Structure;
using System.ComponentModel;
using System.Windows.Controls;
using System.Windows;

namespace UpscalerWPF
{
    class Upscaler
    {
        public int batchSize = 1;
        public int scale { get; set; } = 2;
        public string filePath = "";
        public bool useTensorRT { get; set; } = false;
        public bool useFP16 { get; set; } = false;
        public string modelName { get; set; } = "BSRGAN";
        private string modelPath = "BSRGANx2.onnx";
        public string[] TensorRTSupportModels { get; } = { "BSRGAN" };
        public void setModel(string modelName, int scale)
        {
            this.modelName = modelName;
            this.scale = scale;
            if (modelName == "SwinIR-M")
            {
                modelPath = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.ONNX";
                this.scale = 2;
            }
            else if (modelName == "BSRGAN")
            {
                if (scale == 2)
                {
                    modelPath = "BSRGANx2.onnx";
                }
                else
                {
                    modelPath = "BSRGAN.onnx";
                }
            }
        }
        
        private float clamp(float number, float min, float max)
        {
            if (number < min)
            {
                return min;
            }
            else if (number > max)
            {
                return max;
            }
            else
            {
                return number;
            }
        }
        private Tensor<float> toTensor(Mat[] mat, int height, int width, int batchSize)
        {
            // bgr Mat to rgb normalized tensor 
            // swinIR model option window_size = 8
            int window_size = 8;
            int h_pad = 0;
            int w_pad = 0;
            if (height % window_size > 0)
            {
                h_pad = (height / window_size + 1) * window_size - height;
            }
            if (width % window_size > 0)
            {
                w_pad = (width / window_size + 1) * window_size - width;
            }            
            
            Tensor<float> tensor = new DenseTensor<float>(new[] { batchSize, 3, height+h_pad, width+w_pad });
            
            for (int b = 0; b<batchSize; b++)
            {
                Image<Rgb, byte> tmp = mat[b].ToImage<Rgb, byte>();
                Parallel.For(0, height, y =>
                {
                    for (int x = 0; x < width; x++)
                    {
                        tensor[b, 0, y, x] = tmp.Data[y, x, 0] / 255f;
                        tensor[b, 1, y, x] = tmp.Data[y, x, 1] / 255f;
                        tensor[b, 2, y, x] = tmp.Data[y, x, 2] / 255f;
                    }
                });                
            }            
            return tensor;
        }
        
        private Mat[] toMat(Tensor<float> tensor, int height, int width, int batchSize)
        {
            Mat[] frames = new Mat[batchSize];
            for (int b = 0; b<batchSize; b++)
            {
                var frame = new Mat(height * this.scale, width * this.scale, Emgu.CV.CvEnum.DepthType.Cv8U, 3).ToImage<Rgb, Byte>();                
                Parallel.For(0, height * this.scale, y =>
                {
                    for (int x = 0; x < width * this.scale; x++)
                    {
                        frame.Data[y, x, 0] = (byte)(clamp(tensor[b, 0, y, x], 0, 1) * 255);
                        frame.Data[y, x, 1] = (byte)(clamp(tensor[b, 1, y, x], 0, 1) * 255);
                        frame.Data[y, x, 2] = (byte)(clamp(tensor[b, 2, y, x], 0, 1) * 255);
                    }
                });
                frames[b] = frame.Mat;
            }            
            return frames;
        }

        public bool DoUpscaling(BackgroundWorker worker, DoWorkEventArgs e)
        {
            //OrtEnv.Instance().EnvLogLevel = 0;
            bool success = false;
            OrtEnv.Instance();
            OrtTensorRTProviderOptions options = new OrtTensorRTProviderOptions();
            SessionOptions sessionOptions = new SessionOptions();            
            options.UpdateOptions(new Dictionary<string, string>() {
                { "trt_fp16_enable", useFP16? "true" : "false" }, 
                { "trt_max_partition_iterations", "30" },
                { "trt_engine_cache_enable", "true" }
            });
            
            if (useTensorRT)
            {
                sessionOptions.AppendExecutionProvider_Tensorrt(options);
            }
            sessionOptions.AppendExecutionProvider_CUDA();
            
            var session = new InferenceSession(modelPath, sessionOptions);


            // session warmup
            if (useTensorRT)
            {
                // minimum range
                session.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1, 3, 160, 160 })) });
                // maximum range
                session.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new[] { 1, 3, 720, 1280 })) });
            }

            VideoCapture capture = new VideoCapture(this.filePath);
            
            string savePath = "upscaled_" + Path.GetFileName(this.filePath);
            int width = (int)capture.Get(Emgu.CV.CvEnum.CapProp.FrameWidth);
            int height = (int)capture.Get(Emgu.CV.CvEnum.CapProp.FrameHeight);
            System.Drawing.Size size = new System.Drawing.Size(width*this.scale, height*this.scale);
            double fps = capture.Get(Emgu.CV.CvEnum.CapProp.Fps);
            
            int totalFrame = (int)capture.Get(Emgu.CV.CvEnum.CapProp.FrameCount);
            int backend_idx = 0;
            
            foreach(Backend be in CvInvoke.WriterBackends)
            {
                if (be.Name.Equals("MSMF"))
                {
                    backend_idx = be.ID;
                    break;
                }
            }
            VideoWriter writer = new VideoWriter(savePath, backend_idx, VideoWriter.Fourcc('H', '2', '6', '4'), fps, size, true);            
            Stopwatch stopwatch = new Stopwatch();
            Stopwatch stepwatch = new Stopwatch();
            stopwatch.Start();
           
            while (true)
            {
                if (worker.CancellationPending)
                {
                    e.Cancel = true;                    
                    break;
                }
                else
                {
                    
                    stepwatch.Start();
                    Mat[] frames = new Mat[this.batchSize];
                    int lastBatchIdx = 0;
                    for (int i = 0; i < this.batchSize; i++)
                    {
                        Mat frame = capture.QueryFrame(); // bgr frame
                        if (frame == null) break;
                        lastBatchIdx = i;
                        frames[i] = frame;
                    }
                    var pos = (int)capture.Get(Emgu.CV.CvEnum.CapProp.PosFrames);
                                        
                    
                    Tensor<float> tensor = toTensor(frames, height, width, lastBatchIdx + 1);
                    
                    var inputs = new List<NamedOnnxValue> {NamedOnnxValue.CreateFromTensor("input", tensor)};
                    Tensor<float> output = session.Run(inputs).ToList().First().Value as Tensor<float>;
                    var up_frames = toMat(output, height, width, lastBatchIdx + 1);
                    
                    for (int i = 0; i < lastBatchIdx + 1; i++)
                    {
                        writer.Write(up_frames[i]);
                    }
                    
                    stepwatch.Stop();
                    // finished
                    int futureWork = totalFrame - pos;
                    TimeSpan workingTime = new TimeSpan(stepwatch.ElapsedMilliseconds*10000*futureWork);

                    string format = @"hh\:mm\:ss";                    
                    int percentComplete = (int)((float)pos / (float)totalFrame * 100);
                    String[] time = {stopwatch.Elapsed.ToString(format), workingTime.ToString(format), "Video Converting"};
                    worker.ReportProgress(percentComplete, time);
                    stepwatch.Reset();
                    if (pos == totalFrame)
                    {
                        success = true;
                        break;
                    }
                }                
            }
            capture.Dispose();
            writer.Dispose();            
            session.Dispose();
            stopwatch.Stop();
            return success;
        }
        public void FFmpegConvert()
        {
            string upscaledPath = "upscaled_" + Path.GetFileName(this.filePath);
            Process proc = new Process();
            string cmd = $"-i {filePath} -i {upscaledPath} -map 1:v -map 0:a -c:a copy -c:v libx264 -pix_fmt yuv420p -progress nostat encoded_{Path.GetFileName(this.filePath)}";
            proc.StartInfo.FileName = ".\\ffmpeg\\bin\\ffmpeg.exe";
            proc.StartInfo.Arguments = cmd;
            proc.StartInfo.UseShellExecute = false;
            proc.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            proc.StartInfo.RedirectStandardError = true;
            proc.StartInfo.RedirectStandardOutput = true;
            proc.StartInfo.CreateNoWindow = true;
            var err = proc.Start();
            StreamReader sr = proc.StandardError;
            while (!sr.EndOfStream)
            {
                var line = sr.ReadLine();
                Console.WriteLine(line);
                getTotalSecondProcessed(line);
            }            
        }

        private void getTotalSecondProcessed(string line)
        {
            try
            {
                string[] split = line.Split(" ");
                foreach (var row in split)
                {
                    if (row.StartsWith("time="))
                    {
                        var time = row.Split("=");                        
                    }
                }
            }
            catch { }
        }
        
        public void inference()
        {
            var session = new InferenceSession(modelPath, SessionOptions.MakeSessionOptionWithCudaProvider(0));
            Image<Rgb, Byte> img = new Image<Rgb, byte>(this.filePath);
            // CHW-RGB to NCHW-RGB
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, 720, 1280 });
            for (int y = 0; y < img.Height; y++)
            {                
                for (int x = 0; x < img.Width; x++)
                {
                    input[0, 0, y, x] = img.Data[y, x, 0] / 255f;
                    input[0, 1, y, x] = img.Data[y, x, 1] / 255f;
                    input[0, 2, y, x] = img.Data[y, x, 2] / 255f;
                }
            }
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output = session.Run(inputs);
            var tensor = (output.ToList().First().Value as Tensor<float>);
            var result = new Mat(720*2, 1280*2, Emgu.CV.CvEnum.DepthType.Cv8U, 3).ToImage<Rgb, Byte>();
            
            for (var y = 0; y < 720 * 2; y++)
            {
                for (var x = 0; x < 1280 * 2; x++)
                {
                    result.Data[y, x, 0] = (byte)(tensor[0, 0, y, x]*255);
                    result.Data[y, x, 1] = (byte)(tensor[0, 1, y, x]*255);
                    result.Data[y, x, 2] = (byte)(tensor[0, 2, y, x]*255);
                }
            }
            result.Save("upscaled.png");
        }
    }
}
