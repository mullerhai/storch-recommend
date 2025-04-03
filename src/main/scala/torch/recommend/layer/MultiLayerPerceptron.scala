package torch.recommend.layer

import torch.*
import torch.nn.modules.{HasParams, TensorModule}

class MultiLayerPerceptron[ParamType <: FloatNN: Default](
    input_dim: Int,
    embed_dims: Seq[Int],
    dropout: Float,
    output_layer: Boolean = true
) extends HasParams[ParamType]
    with TensorModule[ParamType] {

  val layers = nn.ModuleList[ParamType]()
  embed_dims.zipWithIndex.foreach {
    case (embed_dim, index) => {
      val block = register(LinearBnReluDropBlock(input_dim, embed_dim, dropout))
      layers.append(block)
    }
  }
  if (output_layer) {
//    var output_linear = register(nn.Linear(input_dim, 1))
    layers.append(register(nn.Linear(input_dim, 1)))
  }
  def apply(input: Tensor[ParamType]): Tensor[ParamType] = {

    layers(input)

  }

}

object MultiLayerPerceptron:
  def apply[ParamType <: FloatNN: Default](
      input_dim: Int,
      embed_dims: Seq[Int],
      dropout: Float,
      output_layer: Boolean = true
  ): MultiLayerPerceptron[ParamType] =
    new MultiLayerPerceptron(input_dim, embed_dims, dropout, output_layer)
//val mlp = nn.Sequential[ParamType]()

//    val moduleDict = nn.ModuleDict[ParamType]()
//          moduleDict.insert("block"+index, block)
