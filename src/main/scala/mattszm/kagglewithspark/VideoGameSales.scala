package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions._


object VideoGameSales {

  case class Sale(Rank: Int, Name: String, Platform: String,
                  Year: String, Genre: String, Publisher: String,
                  NA_Sales: Double, EU_Sales: Double,
                  JP_Sales: Double, Other_Sales: Double)

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
      round($"NA_Sales" + $"EU_Sales" + $"JP_Sales" + $"Other_Sales", 2))

    val salesUntil2016 = salesWithGlobalSales.filter(
      $"Year" =!= "N/A" && $"Year" =!= "2020" && $"Year" =!= "2017")

    val salesGroupedByYear = salesUntil2016.groupBy("Year")
      .agg(round(sum("Global_Sales"), 2)
        .alias("Global_Sales"))

    val sortedSalesGroupedByYear = salesGroupedByYear
      .sort($"Global_Sales".desc)
    val yearWithLargestGlobalSales = sortedSalesGroupedByYear.first()

    val mostPopularTitle = salesWithGlobalSales.filter(
      $"Year" === yearWithLargestGlobalSales(0))
      .select("Name", "Platform", "Publisher").first()

    println(s"In ${yearWithLargestGlobalSales(0)}, there was the largest " +
      s"global sale of ${yearWithLargestGlobalSales(1)} million units.")
    println(s"The most popular title was \'${mostPopularTitle(0)}\' " +
      s"(${mostPopularTitle(1)} version), published by ${mostPopularTitle(2)}.")

  spark.stop()
  }
}
