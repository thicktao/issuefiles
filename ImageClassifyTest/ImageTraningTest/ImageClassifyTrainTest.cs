using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace ImageClassifyTest
{

    public class ImageClassifyTrainTest
    {
        private static readonly string PrePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "train");
        private static readonly string TrainModelPath2 = Path.Combine(PrePath, "data2.zip");
        private static readonly string PreDataPath = Path.Combine(PrePath, "preData.zip");
        private static readonly string DataModelPath = Path.Combine(PrePath, "data.zip");
        private static readonly string InceptionPb = Path.Combine(PrePath, "tensorflow_inception_graph.pb");
        private static readonly string FirstScanDir = Path.Combine(PrePath, "TrainImage1");
        private static readonly string SecondScanDir = Path.Combine(PrePath, "TrainImage2");
        private static readonly string PredictImgs = Path.Combine(PrePath, "PredictImgs/111.png");
        private static readonly MLContext MlContext = new MLContext(1);

        public static void SaveRetrainModel()
        {

            List<ImageData> list1 = new List<ImageData>();
            ScanPic(list1, FirstScanDir);
            var fulldata1 = MlContext.Data.LoadFromEnumerable(list1);
            var trainTestData1 = MlContext.Data.TrainTestSplit(fulldata1);
            var trainingDataView1 = trainTestData1.TrainSet;

            var pipeline = MlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(MlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: ImageSettings.ImageWidth, imageHeight: ImageSettings.ImageHeight, inputColumnName: "Image"))
                .Append(MlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageSettings.ChannelsLast, offsetImage: ImageSettings.Mean))
                .Append(MlContext.Model.LoadTensorFlowModel(InceptionPb).ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                .AppendCacheCheckpoint(MlContext);

            var trainingPipeline = pipeline.Append(MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "softmax2_pre_activation"));

            var dataPiple = trainingPipeline.Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"));

            var preDataTransform = trainingPipeline.Fit(trainingDataView1);
            MlContext.Model.Save(preDataTransform, trainingDataView1.Schema, PreDataPath);

            ITransformer dataTransform = dataPiple.Fit(trainingDataView1);
            MlContext.Model.Save(dataTransform, trainingDataView1.Schema, DataModelPath);

            PredictScore();

        }

        public static void SecondTrainAndPredit()
        {
            var list2 = new List<ImageData>();
            ScanPic(list2, SecondScanDir);
            var fulldata2 = MlContext.Data.LoadFromEnumerable(list2);
            var trainTestData2 = MlContext.Data.TrainTestSplit(fulldata2);
            var trainingDataView2 = trainTestData2.TrainSet;


            var preDataModel = MlContext.Model.Load(PreDataPath, out DataViewSchema modelInputSchema2);
            var originalModelParameters = (preDataModel as TransformerChain<ITransformer>)?.LastTransformer as MulticlassPredictionTransformer<MaximumEntropyModelParameters>;

            ITransformer dataPrepPipeline = MlContext.Model.Load(DataModelPath, out var dataPrepPipelineSchema);
            IDataView newDataForm = dataPrepPipeline.Transform(trainingDataView2);
            var _keyToValueModel = MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "softmax2_pre_activation").Fit(newDataForm, originalModelParameters.Model);

            MlContext.Model.Save(_keyToValueModel, trainingDataView2.Schema, TrainModelPath2);

            PredictScore(TrainModelPath2);
        }

        private static void PredictScore(string dataModelPath = "")
        {
            if (string.IsNullOrEmpty(dataModelPath))
            {
                dataModelPath = DataModelPath;
            }
            var loadedModel = MlContext.Model.Load(dataModelPath, out var modelInputSchema);
            var predictor = MlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);
            var imageData = new ImageData() { Image = (Bitmap)Image.FromFile(PredictImgs) };
            var result = predictor.Predict(imageData);
            Console.WriteLine(result.Score.Max());
        }

        private static void ScanPic(List<ImageData> list, string directory)
        {
            var files = Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories);
            StringBuilder imgTags = new StringBuilder();
            foreach (var filePath in files)
            {
                if (!filePath.EndsWith(".jpg") && !filePath.EndsWith(".png"))
                {
                    continue;
                }
                var deviceModel = Directory.GetParent(filePath).Name;
                string imgPath = $"{deviceModel}/{Path.GetFileName(filePath)}";
                imgTags.AppendLine($"{imgPath}\t{deviceModel}");
                list.Add(new ImageData()
                {

                    Label = deviceModel,
                    Image = (Bitmap)Image.FromFile(filePath)
                });
            }
        }
    }
}
