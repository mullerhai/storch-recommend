//package torch.recommend.dataset;
//
//import scala.Tuple2;
//import scala.util.Random;
//import torch.Device;
//import torch.nn.loss.CrossEntropyLoss;
//import torch.optim.Adam;
//import torch.recommend.dataset.LstmNet;
//import torch.recommend.dataset.GruNet;
//import torch.recommend.dataset.RnnNet;
//import torch.*;
//import torchvision.datasets.FashionMNIST;
//
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.Arrays;
//import java.util.Objects;
//
//public class Hi {
//    public static void main(String[] args) {
//     System.out.println("hi");   
////     LstmNetApp.main(args);
//     Device device = Device.CPU();
//     LstmNet<Float32> model= new LstmNet(28,128,2,10,Default.float32());
//     model.to(device);
//     Path dataPath = Paths.get("D:\\data\\FashionMNIST");
//     FashionMNIST mnistTrain = new FashionMNIST(dataPath, true,  false) ;
//     FashionMNIST mnistEval = new  FashionMNIST(dataPath,  false, false);
////     System.out.println(model.parameters());
//     System.out.println(model.summarize());
//     scala.Tuple2<Object, Object> betas =
//                new scala.Tuple2<>(0.9, 0.999);
//     Adam optimizer = new Adam(model.parameters().collect(), 1e-3d, betas, 1e-8,0, true);
////     CrossEntropyLoss lossFn = new CrossEntropyLoss();
//     Tensor<Float32> evalFeatures = mnistEval.features();
//     evalFeatures.to(device);
//     Tensor<Int64>  evalTargets = mnistEval.targets();
//     evalTargets.to(device);
//     Random random = new Random(123);
//     Tensor<Float32> predictions = model.apply(evalFeatures);//.reshape(Arrays.stream(-1, 28, 28)));
////     random.shuffle(mnistTrain).forEach(batch -> {
////         Tensor<Float32> features = batch.features();
////         features.to(device);
////         Tensor<Int64>  targets = batch.targets();
////         targets.to(device);
////         Tensor<Float32> logits = model(features);
////         Tensor<Float32> loss = lossFn(logits, targets);
////         System.out.println(loss);
////         loss.backward();
//         optimizer.step();
//         optimizer.zeroGrad();
////     });
//    }
//}
