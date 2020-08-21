using System;

namespace ImageClassifyTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("first training being...");
            ImageClassifyTrainTest.SaveRetrainModel();
            Console.WriteLine("first training completed,second training begin ...");
            ImageClassifyTrainTest.SecondTrainAndPredit();
            Console.WriteLine("second training completed");
            Console.ReadKey();
        }
    }
}
