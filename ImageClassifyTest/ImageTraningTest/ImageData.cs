using System.Drawing;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace ImageClassifyTest
{
    public class ImageData
    {
        //[LoadColumn(0)]
        //public string ImagePath;
        [ImageType(227, 227)]
        [LoadColumn(0)]
        public Bitmap Image;

        [LoadColumn(1)]
        public string Label;
    }
}
