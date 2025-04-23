import sbt.*
import Keys.*
import sbt.Def.settings

import scala.collection.immutable.Seq

//lazy val root = project
//  .enablePlugins(NoPublishPlugin)
//  .in(file("."))
////  .aggregate(core, vision, examples, docs)
//  .settings(
//    javaCppVersion := (ThisBuild / javaCppVersion).value,
////    csrCacheDirectory := file("D:\\coursier"),
//  )

ThisBuild / tlBaseVersion := "0.0" // your current series x.y
//ThisBuild / CoursierCache := file("D:\\coursier")
ThisBuild / organization := "io.github.mullerhai" //"dev.storch"
ThisBuild / organizationName := "storch.dev"
ThisBuild / startYear := Some(2024)
ThisBuild / licenses := Seq(License.Apache2)
ThisBuild / developers := List(
  // your GitHub handle and name
  tlGitHubDev("mullerhai", "mullerhai")
)
ThisBuild / version := "0.1.0"

ThisBuild / scalaVersion := "3.6.4"
ThisBuild / tlSonatypeUseLegacyHost := false

import xerial.sbt.Sonatype.sonatypeCentralHost
ThisBuild / sonatypeCredentialHost := sonatypeCentralHost

import ReleaseTransformations._
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  releaseStepCommandAndRemaining("+publishSigned"),
  releaseStepCommandAndRemaining("sonatypeBundleRelease"),
  setNextVersion,
  commitNextVersion,
  pushChanges,
)

lazy val root = (project in file("."))
  .settings(
    name := "storch-recommend",
    javaCppVersion := (ThisBuild / javaCppVersion).value,
  )
ThisBuild / tlSitePublishBranch := Some("main")

ThisBuild / apiURL := Some(new URL("https://storch.dev/api/"))
val scrImageVersion = "4.3.0" //4.0.34
val pytorchVersion =  "2.5.1"// "2.1.2" 2.5.1-1.5.11"
val cudaVersion = "12.6-9.5"  //"12.4.99" // "12.3-8.9"
val openblasVersion = "0.3.28"// "0.3.26"
val mklVersion = "2025.0"//"2024.0"
ThisBuild / scalaVersion := "3.6.2"
ThisBuild / javaCppVersion := "1.5.11"//"1.5.10"
ThisBuild / resolvers ++= Resolver.sonatypeOssRepos("snapshots")

ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("11"))
ThisBuild / githubWorkflowOSes := Seq("macos-latest", "ubuntu-latest", "windows-latest")

val enableGPU = settingKey[Boolean]("enable or disable GPU support")
val hasMKL = {
  val firstPlatform = org.bytedeco.sbt.javacpp.Platform.current.head
  firstPlatform == "linux-x86_64" || firstPlatform == "windows-x86_64"
}
libraryDependencies +=   "dev.storch" % "core_3" % "0.2.3-1.15.1"
libraryDependencies +=   "dev.storch" % "vision_3" % "0.2.3-1.15.1"
libraryDependencies +=  "org.scalameta" %% "munit" % "0.7.29" //% Test
libraryDependencies +=  "org.scalameta" %% "munit-scalacheck" % "0.7.29" // % Test
ThisBuild  / assemblyMergeStrategy := {
  case v if v.contains("module-info.class")   => MergeStrategy.discard
  case v if v.contains("UnusedStub")          => MergeStrategy.first
  case v if v.contains("aopalliance")         => MergeStrategy.first
  case v if v.contains("inject")              => MergeStrategy.first
  case v if v.contains("jline")               => MergeStrategy.discard
  case v if v.contains("scala-asm")           => MergeStrategy.discard
  case v if v.contains("asm")                 => MergeStrategy.discard
  case v if v.contains("scala-compiler")      => MergeStrategy.deduplicate
  case v if v.contains("reflect-config.json") => MergeStrategy.discard
  case v if v.contains("jni-config.json")     => MergeStrategy.discard
  case v if v.contains("git.properties")      => MergeStrategy.discard
  case v if v.contains("reflect.properties")      => MergeStrategy.discard
  case v if v.contains("compiler.properties")      => MergeStrategy.discard
  case v if v.contains("scala-collection-compat.properties")      => MergeStrategy.discard
  case x =>
    val oldStrategy = (assembly / assemblyMergeStrategy).value
    oldStrategy(x)
}