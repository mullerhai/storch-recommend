package torch.recommend.model

import torch.*
import torch.nn.modules.{HasParams, TensorModule}
import torch.recommend.layer as nns
import torch.recommend.layer.IntOrString
import torch.nn.modules.container.{ModuleDict, ModuleList, Sequential}

import scala.collection.mutable.ListBuffer

final class PLEModel[ParamType <: FloatNN : Default](
                                                      categorical_field_dims: Seq[Int],
                                                      numerical_num: Int,
                                                      embed_dim: Int,
                                                      bottom_mlp_dims: Seq[Int],
                                                      tower_mlp_dims: Seq[Int],
                                                      task_num: Int,
                                                      shared_expert_num: Int,
                                                      specific_expert_num: Int,
                                                      dropout: Float
                                                     ) extends HasParams[ParamType]
  with TensorModule[ParamType] {

  val embedding = registerModule(nns.FeaturesEmbedding(categorical_field_dims, embed_dim = embed_dim))
  val numerical_layer = registerModule(nn.Linear(numerical_num, embed_dim))
  val embed_output_dim = (categorical_field_dims.length + 1) * embed_dim
  val layers_num = bottom_mlp_dims.length
  val task_experts = for i <- 0 until(layers_num) yield task_num
  val task_gates = for i <- 0 until(layers_num) yield task_num
  val share_experts =  layers_num
  val share_gates =  layers_num
  for( i <- 0 until(layers_num)){
    val input_dim = if i== 0 then embed_output_dim else bottom_mlp_dims(i-1)
    val shareExpertsLayers = for i <- 0 until(shared_expert_num)yield nns.MultiLayerPerceptron(input_dim,Seq(bottom_mlp_dims(i)),dropout,false)
    share_experts(i)= ModuleList[ParamType]( shareExpertsLayers*)
    share_gates(i) = Sequential[ParamType](nn.Linear(input_dim,shared_expert_num+task_num*specific_expert_num),
      nn.Softmax(dim=1))
    for( j <- 0 until(task_num)){
      val taskExpertsLayer = for k <- 0 until(specific_expert_num) yield nns.MultiLayerPerceptron(input_dim,Seq(bottom_mlp_dims(i)),dropout,false)
      task_experts(i)(j) = ModuleList[ParamType](taskExpertsLayer* )
      task_gates(i)(j) = Sequential[ParamType](
        nn.Linear(input_dim,shared_expert_num+specific_expert_num),
        nn.Softmax(dim= 1)
      )
    }
    task_experts(i) = ModuleList[ParamType](task_experts(i))
    task_gates(i) = ModuleList[ParamType](task_gates(i))
  }
  task_experts = ModuleList[ParamType](task_experts)
  task_gates = ModuleList[ParamType](task_gates)
  share_experts = ModuleList[ParamType](share_experts)
  share_gates = ModuleList[ParamType](share_gates)
  val towerLayers = for i <- 0 until(task_num) yield nns.MultiLayerPerceptron(bottom_mlp_dims.last,tower_mlp_dims,dropout)
  val tower = ModuleList[ParamType](towerLayers*)
  

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???

  def apply(categorical_x: Tensor[ParamType], numerical_x: Tensor[ParamType]): Seq[Tensor[ParamType]] = {
    val categorical_emb = embedding(categorical_x)
    val numerical_emb = numerical_layer(numerical_x).unsqueeze(1)
    val emb = torch.cat(Seq(categorical_emb, numerical_emb), dim = 1).view(-1, embed_output_dim)
    var task_feaSeq = for i <- 0 until(task_num +1) yield emb
    var task_fea = new ListBuffer[Tensor[ParamType]]()
    task_fea.addAll(task_feaSeq)
    for (i <- 0 until(layers_num)){
      var share_output = share_experts(i).map(expert => expert(task_fea(-1)).unsqueeze(1))
      val task_output_list = new ListBuffer[]()
      for(j <- 0 until(task_num)){
        val task_output = task_experts(i)(j).map(expert => expert(task_fea(j)).unsqueeze(1))
        task_output_list.appended(task_output)
        val mix_output = torch.cat(Seq(task_output,share_output),dim=1)
        val gate_value = task_gates(i)(j)(task_fea(j)).unsqueeze(1)
        task_fea(j) = torch.bmm(gate_value,mix_output).squeeze(1)
      }
      if(i != layers_num-1){
        val gate_value = share_gates(i)(task_fea(-1)).unsqueeze(1)
        val mix_output = torch.cat(Seq(task_output_list,share_output),dim =1 )
        task_fea(-1) = torch.bmm(gate_value,mix_output).squeeze(1)
      }
    }
    val results = for i <- 0 until(task_num) yield torch.sigmoid(tower(i)(task_fea(i)).squeeze(1))
    results
  }
}

