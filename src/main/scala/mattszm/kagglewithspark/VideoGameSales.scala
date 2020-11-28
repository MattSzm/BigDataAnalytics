package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions._


object VideoGameSales {

  case class Sale(Rank: Int, Name: String, Platform: String,
                  Year: String, Genre: String, Publisher: String,
                  NA_Sales: Double, EU_Sales: Double,
                  JP_Sales: Double, Other_Sales: Double)

  def mostProfitableYear(spark: SparkSession, data: Dataset[Sale]): Unit = {
    import spark.implicits._

    val salesWithGlobalSales = data.withColumn(
      "Global_Sales",
      round($"NA_Sales" + $"EU_Sales" + $"JP_Sales" + $"Other_Sales", 2))

    val salesUntil2016 = salesWithGlobalSales.filter(
      $"Year" =!= "N/A" && $"Year" =!= "2020" && $"Year" =!= "2017")

    val salesGroupedByYear = salesUntil2016.groupBy("Year")
      .agg(round(sum("Global_Sales"), 2)
        .alias("Global_Sales"))
    val sortedSalesGroupedByYear = salesGroupedByYear
      .sort($"Global_Sales".desc)
    val yearWithLargestGlobalSales = sortedSalesGroupedByYear.first()

    // salesWithGlobalSales are already sorted
    val mostPopularTitle = salesWithGlobalSales.filter(
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

    mostProfitableYear(spark, data=salesRaw)
    genreWithRegion(spark, data=salesRaw)
    genreWithRegionSplitByPlatform(spark,data=salesRaw)

    spark.stop()
  }
}
