using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace UpscalerWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Upscaler upscaler;
        private BackgroundWorker inferenceWorker;
        public MainWindow()
        {            
            InitializeComponent();
            InitializeBackgroundWorker();
            OrtEnv.Instance();
        }
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {            
            upscaler = new Upscaler();            
        }
        private void InitializeBackgroundWorker()
        {   
            inferenceWorker = new BackgroundWorker();
            inferenceWorker.WorkerSupportsCancellation = true;
            inferenceWorker.WorkerReportsProgress = true;
            inferenceWorker.DoWork += new DoWorkEventHandler(inferenceWorker_DoWork);
            inferenceWorker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(inferenceWorker_RunWorkerCompleted);
            inferenceWorker.ProgressChanged += new ProgressChangedEventHandler(inferenceWorker_ProgressChanged);
        }

        private void inferenceWorker_DoWork(object sender, DoWorkEventArgs e)
        {
            // Get the BackgroundWorker that raised this event.
            BackgroundWorker worker = sender as BackgroundWorker;
            this.upscaler.DoUpscaling(worker, e);
            //this.upscaler.demuxing(worker, e);
        }
        private void inferenceWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            if (e.Error != null)
            {
                MessageBox.Show(e.Error.Message);
            }
            else if (e.Cancelled)
            {                
                state.Text = "Canceled";
            }
            else
            {
                // succeeded.
                state.Text = "Complited";
            }

            // Enable the UpDown control.
            batchSize.IsEnabled = true;

            // Enable the Start button.
            startBtn.IsEnabled = true;

            // Disable the Cancel button.
            cancelBtn.IsEnabled = false;
        }
        private void inferenceWorker_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            String[] str = (String[])e.UserState;
            progressBar.Value = e.ProgressPercentage;
            state.Text = str[2] + "  " +e.ProgressPercentage + "%";
            
            this.estimatedTime.Text = "EstimatedTime : " + str[1];
            this.totalTime.Text = "Total : " + str[0];
        }

        private void openBtn_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog();                        
            dialog.Filter = "Video File(.mp4)|*.mp4"; // Filter files by extension
            bool? result = dialog.ShowDialog();
            if (result == true)
            {
                // Save document
                string filename = dialog.FileName;
                path.Text = filename;
                upscaler.filePath = filename;
            }
        }

        private void startBtn_Click(object sender, RoutedEventArgs e)
        {
            state.Text = "Initializing";

            this.batchSize.IsEnabled = false;
            this.startBtn.IsEnabled = false;
            this.cancelBtn.IsEnabled = true;

            // Start the asynchronous operation.
            inferenceWorker.RunWorkerAsync();
        }

        private void cancelBtn_Click(object sender, RoutedEventArgs e)
        {
            this.inferenceWorker.CancelAsync();

            // Disable the Cancel button.
            cancelBtn.IsEnabled = false;
        }

        private void modelSelect_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (upscaler == null) return;
            var item = (ComboBoxItem)modelSelect.SelectedItem;
            var modelName = item.Content.ToString();
            bool isTensorRTSupportedModel = Array.Exists(upscaler.TensorRTSupportModels, model => model == modelName);

            upscaler.modelName = modelName;
            upscaler.setModel(upscaler.modelName, upscaler.scale);
            
            if (isTensorRTSupportedModel)
            {
                isTensorRT.IsEnabled = true;                
            }
            else {
                isTensorRT.IsEnabled = false;
                isTensorRT.IsChecked = false;
            }
            
        }

        private void radio2x_Checked(object sender, RoutedEventArgs e)
        {
            if (upscaler == null) return;
            upscaler.scale = 2;
            upscaler.setModel(upscaler.modelName, upscaler.scale);
        }

        private void radio4x_Checked(object sender, RoutedEventArgs e)
        {
            if (upscaler == null) return;
            upscaler.scale = 4;
            upscaler.setModel(upscaler.modelName, upscaler.scale);
        }

        private void isTensorRT_Checked(object sender, RoutedEventArgs e)
        {
            upscaler.useTensorRT = (bool)isTensorRT.IsChecked;
            isFP16.IsEnabled = true;
        }

        private void isTensorRT_Unchecked(object sender, RoutedEventArgs e)
        {
            upscaler.useTensorRT = (bool)isTensorRT.IsChecked;
            isFP16.IsChecked = false;
            isFP16.IsEnabled = false;
        }

        private void isFP16_Checked(object sender, RoutedEventArgs e)
        {
            upscaler.useFP16 = (bool)isFP16.IsChecked;
        }

        private void isFP16_Unchecked(object sender, RoutedEventArgs e)
        {
            upscaler.useFP16 = (bool)isFP16.IsChecked;
        }
    }
}
