

import torch.recommend.model.{SingleTaskModel,SharedBottomModel,PLEModel,OMoEModel,MMoEModel,AITMModel ,FieldAwareNeuralFactorizationMachineModel,FieldAwareFactorizationMachineModel,NeuralFactorizationMachineModel,NeuralCollaborativeFiltering,LogisticRegressionModel,ProductNeuralNetworkModel ,ExtremeDeepFactorizationMachineModel ,AdaptiveFactorizationNetwork,AttentionalFactorizationMachineModel,AutomaticFeatureInteractionModel,HighOrderFactorizationMachineModel  ,DSSM,WideAndDeepModel,DeepCrossNetworkModel,FactorizationSupportedNeuralNetworkModel ,FactorizationMachineModel,DeepFactorizationMachineModel}

object testModel {

  def main(args: Array[String]): Unit = {

    //    user_datatypes,
    //    item_datatypes,
    //    user_dnn_size,
    //    item_dnn_size,
    //    dropout,
    //    activation,
    //    use_senet
    //todo  not really work need fixup java.lang.RuntimeException: from is out of bounds for float
//    val model = DSSM(
//      user_datatypes = Seq(Map("user_id" -> 10)),
//      item_datatypes = Seq(Map("item_id" -> 10)),
////      user_dnn_size = Seq(10, 10, 10),
////      item_dnn_size = Seq(10, 10, 10),
//      dropout = 0.2f,
//      activation = "relu",
//      use_senet = false
//    )
    //todo work 22
    val model = OMoEModel(
      categorical_field_dims = Seq(10, 10, 10),
      numerical_num = 10,
      embed_dim = 10,
      bottom_mlp_dims = Seq(10, 10, 10),
      tower_mlp_dims = Seq(10, 10, 10),
      task_num = 10,
      expert_num = 10,
      dropout = 0.2f
    )

    //todo work 21
//    val model = MMoEModel(
//      categorical_field_dims = Seq(10, 10, 10),
//      numerical_num = 10,
//      embed_dim = 10,
//      bottom_mlp_dims = Seq(10, 10, 10),
//      tower_mlp_dims = Seq(10, 10, 10),
//      task_num = 10,  // 任务数量
//      expert_num = 10,  // 专家数量
//      dropout = 0.2f
//    )
    //todo work 20
//    val model = AITMModel(
//      categorical_field_dims = Seq(10, 10, 10),
//      numerical_num = 10,
//      embed_dim = 10,
//      bottom_mlp_dims = Seq(10, 10, 10),
//      tower_mlp_dims = Seq(10, 10, 10),
//      task_num = 10,
//      dropout = 0.2f
//    )
    //todo work 19
//    val model = PLEModel(
//      categorical_field_dims = Seq(10, 10, 10),
//      numerical_num = 10,
//      embed_dim = 10,
//      bottom_mlp_dims = Seq(10, 10, 10),
//      tower_mlp_dims = Seq(10, 10, 10),
//      task_num = 10,
//      shared_expert_num = 10,
//      specific_expert_num = 10,
//      dropout = 0.2f
//    )
    //todo work 18
//    val model = SharedBottomModel(
//      categorical_field_dims = Seq(10, 10, 10),
//      numerical_num = 10,
//      embed_dim = 10,
//      bottom_mlp_dims = Seq(10, 10, 10),
//      tower_mlp_dims = Seq(10, 10, 10),
//      task_num = 10,
//      dropout = 0.2f
//    )
    //todo work 17
//    val model = SingleTaskModel(categorical_field_dims = Seq(10, 10, 10), numerical_num = 10, embed_dim = 10, bottom_mlp_dims = Seq(10, 10, 10), tower_mlp_dims = Seq(10, 10, 10), task_num = 10, dropout = 0.2f)

    //todo work 16
//    //field_dims, user_field_idx, item_field_idx,embed_dim, mlp_dims, dropout
//    val model = NeuralCollaborativeFiltering(
//      field_dims = Seq(10, 10, 10),
//      user_field_idx = 0,
//      item_field_idx = 1,
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropout = 0.2
//    )

    //todo work 15
    //(field_dims, embed_dim, mlp_dims, dropouts
//    val model = NeuralFactorizationMachineModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropouts = Seq(0.2f, 0.2f, 0.2f)
//    )
    //todo work 14
//    val model = LogisticRegressionModel(
//      field_dims = Seq(10, 10, 10)
//    )
    //todo work 13
//    val model = ProductNeuralNetworkModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropout = 0.2
//    )
    //todo work 12
//    val model = FieldAwareFactorizationMachineModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10
//    )

    //todo work 11
    //field_dims, embed_dim, mlp_dims, dropouts
//    val model = FieldAwareNeuralFactorizationMachineModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropouts = Seq(0.2f, 0.2f, 0.2f)
//    )
    //todo work 10 ,maybe need to fixup
    //field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half)
//    val model = ExtremeDeepFactorizationMachineModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropout = 0.2f,
//      cross_layer_sizes = Seq(10, 10, 10),
//      split_half = true
//    )
    //todo work 1
//    val model = FactorizationMachineModel(Seq(10, 10, 10), 10)
    //todo work 2
//    val model =DeepFactorizationMachineModel(
//          field_dims = Seq(10, 10, 10),
//          embed_dim = 10,
//          mlp_dims = Seq(10, 10, 10),
//          dropout = 0.2,
//        )
    //todo work 3
//    val model = AdaptiveFactorizationNetwork(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      LNN_dim = 4,
//      dropouts = Seq(0.2f, 0.2f, 0.2f)
//    )
    //todo  work 4
//    val model = WideAndDeepModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropout = 0.2
//    )

    //todo work 5
//    val model = FactorizationSupportedNeuralNetworkModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropout = 0.2
//    )

    //todo work 6
//    val model = DeepCrossNetworkModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      num_layers = 3,
//      mlp_dims = Seq(10, 10, 10),
//      dropout = 0.2
//    )
    //todo work 7
    //    val model = HighOrderFactorizationMachineModel(
    //      field_dims = Seq(10, 10, 10),
    //      embed_dim = 10, order= 2,
    //      mlp_dims = Seq(10, 10, 10),
    //      dropout = 0.2,
    //    )
    // field_dims, embed_dim, atten_embed_dim, num_heads, num_layers, mlp_dims, dropouts,has_residual
    //todo work 8
//    val model = AutomaticFeatureInteractionModel(
//      field_dims = Seq(10, 10, 10),
//      atten_embed_dim = 10,
//      num_heads = 2,
//      embed_dim = 10,
//      mlp_dims = Seq(10, 10, 10),
//      dropouts = Seq(0.2f,0.2f,0.2f),
//      num_layers = 2,
//      has_residual = true
//    )

    //todo work 9
   // field_dims, embed_dim, attn_size, dropouts
//    val model = AttentionalFactorizationMachineModel(
//      field_dims = Seq(10, 10, 10),
//      embed_dim = 10,
//      attn_size = 10,
//      dropouts = Seq(0.2f,0.2f,0.2f),
//    )
//    val model = DSSM(
//
//      dropout = 0.2,
//      task_num = 2
//    )



    println(model)

    val categorical_x = torch.randint(0, 10, Seq(10, 3))
    val numerical_x = torch.rand(10, 10)
    val y = torch.rand(10)
//    val optimizer = torch.optim.Adam(model.parameters(true), lr = 0.001)
    println("Start training...")
  }
}
