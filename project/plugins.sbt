addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.5.2")
addSbtPlugin("org.bytedeco" % "sbt-javacpp" % "1.17")
addSbtPlugin("org.scalameta" % "sbt-mdoc" % "2.5.2")
addSbtPlugin("com.github.sbt" % "sbt-unidoc" % "0.5.0")
addSbtPlugin("org.typelevel" % "sbt-typelevel" % "0.6.5")
addSbtPlugin("org.typelevel" % "sbt-typelevel-site" % "0.6.5")
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.3.0")
addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.12.2")
addSbtPlugin("com.github.sbt" % "sbt-pgp" % "2.1.2")
addSbtPlugin("com.github.sbt" % "sbt-ci-release" % "1.9.3")
addSbtPlugin("com.github.sbt" % "sbt-release" % "1.4.0")

if (sys.env.isDefinedAt("GITHUB_ACTION")) {
  Def.settings(
    addSbtPlugin("net.virtual-void" % "sbt-hackers-digest" % "0.1.2")
  )
} else {
  Nil
}
//addSbtPlugin("io.get-coursier" % "sbt-coursier" % "2.1.22")