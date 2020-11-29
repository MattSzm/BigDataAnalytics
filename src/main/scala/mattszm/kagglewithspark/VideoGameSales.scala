package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoder,
  StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor,
  RandomForestRegressor}


object VideoGameSales {

  case class Sale(Rank: Int, Name: String, Platform: String,
                  Year: String, Genre: String, Publisher: String,
                  NA_Sales: Double, EU_Sales: Double,
                  JP_Sales: Double, Other_Sales: Double)

  case class SaleWithGlobal(Rank: Int, Name: String, Platform: String,
                  Year: String, Genre: String, Publisher: String,
                  NA_Sales: Double, EU_Sales: Double,
                  JP_Sales: Double, Other_Sales: Double,
                  Global_Sales: Double)

  def mostProfitableYear(spark: SparkSession, data: Dataset[SaleWithGlobal]): Unit = {
    import spark.implicits._

    val salesGroupedByYear = data.groupBy("Year")
      .agg(round(sum("Global_Sales"), 2)
        .alias("Global_Sales"))
    val sortedSalesGroupedByYear = salesGroupedByYear
      .sort($"Global_Sales".desc)
    val yearWithLargestGlobalSales = sortedSalesGroupedByYear.first()

    // salesWithGlobalSales are already sorted
    val mostPopularTitle = data.filter(
      $"Year" === yearWithLargestGlobalSales(0))
      .select("Name", "Platform", "Publisher").first()

    println(s"In ${yearWithLargestGlobalSales(0)}, there was the largest " +
      s"global sale of ${yearWithLargestGlobalSales(1)} million units.")
    println(s"The most popular title was \'${mostPopularTitle(0)}\' " +
      s"(${mostPopularTitle(1)} version), published by ${mostPopularTitle(2)}.")
  }

  def genreWithRegion(spark: SparkSession, data: Dataset[Sale]): Unit = {
    import spark.implicits._

    val gamesGroupedByGenre = data.groupBy("Genre")
      .agg(round(sum("NA_Sales"), 2).alias("NA_Sales"),
        round(sum("EU_Sales"), 2).alias("EU_Sales"),
        round(sum("JP_Sales"), 2).alias("JP_Sales"),
        round(sum("Other_Sales"), 2).alias("Other_Sales"))

    val genresWithBestSales = gamesGroupedByGenre.withColumn("Best_Sales",
      array_max(array("NA_Sales", "EU_Sales",
        "JP_Sales", "Other_Sales")))
    val genresWithBestSalesAndRegions = genresWithBestSales.withColumn(
      "Best_Region",
      when(col("Best_Sales") === $"Na_Sales", "Na")
        .otherwise(when(col("Best_Sales") === $"EU_Sales", "EU")
          .otherwise(when(col("Best_Sales") === $"JP_Sales", "JP")
            .otherwise("Other")))
    )

    val sizeOfOutput = genresWithBestSalesAndRegions.count().toInt
    println("\nAnalysis based on determining in which region a given " +
      "genre was the most popular/profitable.")
    genresWithBestSalesAndRegions.show(sizeOfOutput)
  }

  def genreWithRegionSplitByPlatform(spark: SparkSession, data: Dataset[Sale]): Unit = {
    import spark.implicits._

    val gamesGroupedByGenreAndPlatform = data.groupBy("Genre", "Platform")
      .agg(round(sum("NA_Sales"), 2).alias("NA_Sales"),
        round(sum("EU_Sales"), 2).alias("EU_Sales"),
        round(sum("JP_Sales"), 2).alias("JP_Sales"),
        round(sum("Other_Sales"), 2).alias("Other_Sales"))

    val genresWithBestSales = gamesGroupedByGenreAndPlatform.withColumn(
      "Best_Sales", array_max(array("NA_Sales", "EU_Sales",
        "JP_Sales", "Other_Sales")))
    val genresWithBestSalesAndRegions = genresWithBestSales.withColumn(
      "Best_Region",
      when(col("Best_Sales") === $"Na_Sales", "Na")
        .otherwise(when(col("Best_Sales") === $"EU_Sales", "EU")
          .otherwise(when(col("Best_Sales") === $"JP_Sales", "JP")
            .otherwise("Other")))
    ).cache()

    val genresWithBestSalesAndRegionsSorted = genresWithBestSalesAndRegions
      .sort("Genre")

    val sizeOfOutput = genresWithBestSalesAndRegionsSorted.count().toInt
    println("\nAnalysis based on determining in which region a given " +
      "genre was the most popular/profitable, split by platform.")
    println(s"Data split by $sizeOfOutput parts.")
    genresWithBestSalesAndRegionsSorted.show(sizeOfOutput)


    val bestSalesForPlatform = genresWithBestSalesAndRegions
      .groupBy("Platform")
      .agg(max("Best_Sales").alias("Best_Sales"))
    val bestSalesForPlatformWithGenre = bestSalesForPlatform.as("bestSalesForPlatform")
      .join(genresWithBestSalesAndRegions.as("genresWithBestSalesAndRegions"),
        $"bestSalesForPlatform.Platform" === $"genresWithBestSalesAndRegions.Platform" &&
        $"bestSalesForPlatform.Best_Sales" === $"genresWithBestSalesAndRegions.Best_Sales")
      .repartition(100)

    val bestSalesForPlatformWithGenreSorted = bestSalesForPlatformWithGenre
      .sort("bestSalesForPlatform.Platform")
      .select("bestSalesForPlatform.Platform",
        "genresWithBestSalesAndRegions.Genre",
        "bestSalesForPlatform.Best_Sales",
        "genresWithBestSalesAndRegions.Best_Region"
      )

    val sizeOfGenres = bestSalesForPlatformWithGenreSorted.count().toInt
    println("\nAnalysis based on determining which genre " +
      "and in which region was most popular/profitable on " +
      "particular platform")
    bestSalesForPlatformWithGenreSorted.show(sizeOfGenres)
  }

  def mlPrediction(data: Dataset[SaleWithGlobal]): Unit = {
    println("\nUsing machine learning to predict global sale")

    val platformIndexer = new StringIndexer()
      .setInputCol("Platform")
      .setOutputCol("PlatformIndex")
      .setHandleInvalid("skip")
    val yearIndexer = new StringIndexer()
      .setInputCol("Year")
      .setOutputCol("YearIndex")
      .setHandleInvalid("skip")
    val genreIndexer = new StringIndexer()
      .setInputCol("Genre")
      .setOutputCol("GenreIndex")
      .setHandleInvalid("skip")
    val publisherIndexer = new StringIndexer()
      .setInputCol("Publisher")
      .setOutputCol("PublisherIndex")
      .setHandleInvalid("skip")

    val platformEncoder = new OneHotEncoder()
      .setInputCol("PlatformIndex")
      .setOutputCol("PlatformVec")
    val yearEncoder = new OneHotEncoder()
      .setInputCol("YearIndex")
      .setOutputCol("YearVec")
    val genreEncoder = new OneHotEncoder()
      .setInputCol("GenreIndex")
      .setOutputCol("GenreVec")
    val publisherEncoder = new OneHotEncoder()
      .setInputCol("PublisherIndex")
      .setOutputCol("PublisherVec")

    val assembler = new VectorAssembler()
      .setInputCols(Array("PlatformVec", "YearVec", "GenreVec",
      "PublisherVec"))
      .setOutputCol("features")

    val trainTestSet = data.randomSplit(Array(0.8, 0.2), 42)
    val trainingSet = trainTestSet(0)
    val testSet = trainTestSet(1)
    val evaluator = new RegressionEvaluator()
      .setLabelCol("Global_Sales")
      .setPredictionCol("prediction")
      .setMetricName("rmse")


    val gbtRegressor = new GBTRegressor()
      .setFeaturesCol("features")
      .setLabelCol("Global_Sales")
      .setMaxIter(10)
    val gbtPipeline = new Pipeline().setStages(Array(
      platformIndexer, yearIndexer, genreIndexer,
      publisherIndexer, platformEncoder, yearEncoder,
      genreEncoder, publisherEncoder, assembler,
      gbtRegressor
    ))
    val gbtModel = gbtPipeline.fit(trainingSet)
    val gbtPredictions = gbtModel.transform(testSet)
    val gbtPredictionsAndGlobalSales = gbtPredictions.select(
      "prediction",
      "Global_Sales")

    val gbtRmse = evaluator.evaluate(gbtPredictionsAndGlobalSales)
    println(s"\nRoot Mean Squared Error (RMSE) of Gradient-boosted " +
      s"tree regression on test data = $gbtRmse")
    var bestRmse = (gbtRmse, "gbt")


    val rfRegressor = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("Global_Sales")
    val rfPipeline = new Pipeline().setStages(Array(
      platformIndexer, yearIndexer, genreIndexer,
      publisherIndexer, platformEncoder, yearEncoder,
      genreEncoder, publisherEncoder, assembler,
      rfRegressor
    ))
    val rfModel = rfPipeline.fit(trainingSet)
    val rfPredictions = rfModel.transform(testSet)
    val rfPredictionsAndGlobalSales = rfPredictions.select(
      "prediction",
      "Global_Sales")

    val rfRmse = evaluator.evaluate(rfPredictionsAndGlobalSales)
    println(s"\nRoot Mean Squared Error (RMSE) of Random forest" +
      s" regression on test data = $rfRmse")
    if (rfRmse < bestRmse._1){
      bestRmse = (rfRmse, "rf")
    }


    bestRmse match {
      case (_, "gbt") =>
        println("\nThe best result was achieved with " +
          "Gradient-boosted tree")
        gbtPredictionsAndGlobalSales.show(
          gbtPredictionsAndGlobalSales.count().toInt
        )
      case (_, "rf") =>
        println("\nThe best result was achieved with " +
          "Random forest regression")
        rfPredictionsAndGlobalSales.show(
          rfPredictionsAndGlobalSales.count().toInt
        )
    }
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("VideoGameSales")
      .master("local[*]")
      .getOrCreate()

    println("Loading data...\n")

    import spark.implicits._
    val salesRaw = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/video_game_sales.csv")
      .as[Sale]

    println("Processing...\n")

    val salesWithGlobalSales = salesRaw.withColumn(
      "Global_Sales",
      round($"NA_Sales" + $"EU_Sales" + $"JP_Sales"
        + $"Other_Sales", 2))
      .as[SaleWithGlobal]

    mostProfitableYear(spark, data=salesWithGlobalSales)
    genreWithRegion(spark, data=salesRaw)
    genreWithRegionSplitByPlatform(spark,data=salesRaw)
    mlPrediction(data=salesWithGlobalSales)

    spark.stop()
  }
}
