package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{GBTRegressor, RandomForestRegressor}


object VideoGameSales {
  val spark: SparkSession = SparkSession
    .builder
    .appName("VideoGameSales")
    .master("local[*]")
    .getOrCreate()

  case class Sale(Rank: Int, Name: String, Platform: String,
                  Year: String, Genre: String, Publisher: String,
                  NA_Sales: Double, EU_Sales: Double,
                  JP_Sales: Double, Other_Sales: Double)

  case class SaleWithGlobal(Rank: Int, Name: String, Platform: String,
                  Year: String, Genre: String, Publisher: String,
                  NA_Sales: Double, EU_Sales: Double,
                  JP_Sales: Double, Other_Sales: Double,
                  Global_Sales: Double)

  def createSalesWithGlobalSales(data: Dataset[Sale]): Dataset[SaleWithGlobal] = {
    import spark.implicits._
    data.withColumn("Global_Sales",
      round($"NA_Sales" + $"EU_Sales" + $"JP_Sales" + $"Other_Sales", 2))
    .as[SaleWithGlobal]
  }

  def mostProfitableYear(data: Dataset[SaleWithGlobal]): Unit = {
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

  def genreWithRegion(data: Dataset[Sale]): Unit = {
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

  def genreWithRegionSplitByPlatform(data: Dataset[Sale]): Unit = {
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
    val platformIndexer = createStringIndexer("Platform", "PlatformIndex")
    val yearIndexer = createStringIndexer("Year", "YearIndex")
    val genreIndexer = createStringIndexer("Genre", "GenreIndex")
    val publisherIndexer = createStringIndexer("Publisher", "PublisherIndex")

    val platformEncoder = createOneHotEncoder("PlatformIndex", "PlatformVec")
    val yearEncoder = createOneHotEncoder("YearIndex", "YearVec")
    val genreEncoder = createOneHotEncoder("GenreIndex", "GenreVec")
    val publisherEncoder = createOneHotEncoder("PublisherIndex", "PublisherVec")

    val assembler = new VectorAssembler()
      .setInputCols(Array("PlatformVec", "YearVec", "GenreVec",
      "PublisherVec"))
      .setOutputCol("features")
    val stages = Array(
      platformIndexer, yearIndexer, genreIndexer,
      publisherIndexer, platformEncoder, yearEncoder,
      genreEncoder, publisherEncoder, assembler,
    )
    val evaluator = new RegressionEvaluator()
      .setLabelCol("Global_Sales")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val (trainingSet, testSet) = prepareData(data)
    val (rfPredictionsAndGlobalSales, rfRmse) = randomForestRegressorProcessing(
      stages, evaluator, trainingSet, testSet)
    val (gbtPredictionsAndGlobalSales, gbtRmse) = gradientBoostedTreeRegression(
      stages, evaluator, trainingSet, testSet)


    if (gbtRmse > rfRmse) {
      println("\nThe best result was achieved with " +
        "Gradient-boosted tree")
      gbtPredictionsAndGlobalSales.show(
        gbtPredictionsAndGlobalSales.count().toInt)
    }
    else {
      println("\nThe best result was achieved with " +
        "Random forest regression")
      rfPredictionsAndGlobalSales.show(
        rfPredictionsAndGlobalSales.count().toInt)
    }
  }

  def randomForestRegressorProcessing(stages: Array[_<:PipelineStage],
                                      evaluator: RegressionEvaluator,
                                      trainingSet: Dataset[SaleWithGlobal],
                                      testSet: Dataset[SaleWithGlobal]):
                                      (DataFrame, Double) = {
    val rfRegressor = new RandomForestRegressor()
      .setFeaturesCol("features")
      .setLabelCol("Global_Sales")
    val rfStages = stages :+ rfRegressor

    val rfPipeline = new Pipeline().setStages(rfStages)
    val rfModel = rfPipeline.fit(trainingSet)
    val rfPredictions = rfModel.transform(testSet)
    val rfPredictionsAndGlobalSales = rfPredictions.select(
      "prediction",
      "Global_Sales")

    val rfRmse = evaluator.evaluate(rfPredictionsAndGlobalSales)
    println(s"\nRoot Mean Squared Error (RMSE) of Random forest" +
      s" regression on test data = $rfRmse")
    (rfPredictionsAndGlobalSales, rfRmse)
  }

  def gradientBoostedTreeRegression(stages: Array[_<:PipelineStage],
                                    evaluator: RegressionEvaluator,
                                    trainingSet: Dataset[SaleWithGlobal],
                                    testSet: Dataset[SaleWithGlobal]):
                                    (DataFrame, Double) = {
    val gbtRegressor = new GBTRegressor()
      .setFeaturesCol("features")
      .setLabelCol("Global_Sales")
      .setMaxIter(10)
    val gbtStages = stages :+ gbtRegressor

    val gbtPipeline = new Pipeline().setStages(gbtStages)
    val gbtModel = gbtPipeline.fit(trainingSet)
    val gbtPredictions = gbtModel.transform(testSet)
    val gbtPredictionsAndGlobalSales = gbtPredictions.select(
      "prediction",
      "Global_Sales")

    val gbtRmse = evaluator.evaluate(gbtPredictionsAndGlobalSales)
    println(s"\nRoot Mean Squared Error (RMSE) of Gradient-boosted " +
      s"tree regression on test data = $gbtRmse")
    (gbtPredictionsAndGlobalSales, gbtRmse)
  }

  def createStringIndexer(inputCol: String, outputCol: String): StringIndexer =
    new StringIndexer().
      setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setHandleInvalid("skip")

  def createOneHotEncoder(inputCol: String, outputCol: String): OneHotEncoder =
    new OneHotEncoder().
      setInputCol(inputCol)
      .setOutputCol(outputCol)

  def prepareData(data: Dataset[SaleWithGlobal]):
    (Dataset[SaleWithGlobal], Dataset[SaleWithGlobal]) = {
    val trainTestSet = data.randomSplit(Array(0.8, 0.2), 42)
    (trainTestSet(0), trainTestSet(1))
  }

  def main(args: Array[String]): Unit = {
    import spark.implicits._
    Logger.getLogger("org").setLevel(Level.ERROR)
    println("Loading data...\n")

    val salesRaw = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/video_game_sales.csv")
      .as[Sale]

    println("Processing...\n")

    val salesWithGlobalSales = createSalesWithGlobalSales(salesRaw).cache()

    mostProfitableYear(data=salesWithGlobalSales)
    genreWithRegion(data=salesRaw)
    genreWithRegionSplitByPlatform(data=salesRaw)
    mlPrediction(data=salesWithGlobalSales)

    spark.stop()
  }
}
