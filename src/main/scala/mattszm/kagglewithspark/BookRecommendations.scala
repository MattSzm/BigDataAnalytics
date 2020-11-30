package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions._


object BookRecommendations {

  case class Rating(UserID: String, ISBN: String, BookRating: Int)

  case class Book(ISBN: String, BookTitle: String, BookAuthor: String,
                  YearOfPublication: Int, Publisher: String, ImageURLS: String,
                  ImageURLM: String, ImageURLL: String)

  case class User(UserID: String, Location: String, Age: String)

  case class RatingPair(ISBN1: String, ISBN2: String, Rating1:Int,
                        Rating2: Int, UserID: String)

  case class PairSimilarity(ISBN1: String, ISBN2: String,
                            similarity: Double, pairsNum: BigInt)

  def computeCosineSimilarity(spark: SparkSession, data: Dataset[RatingPair])
  : Dataset[PairSimilarity] = {
    val pairScores = data
      .withColumn("xx", col("Rating1") * col("Rating1"))
      .withColumn("yy", col("Rating2") * col("Rating2"))
      .withColumn("xy", col("Rating1") * col("Rating2"))

    val pairSimilarity = pairScores.groupBy("ISBN1", "ISBN2")
      .agg(
        sum(col("xy")).alias("num"),
        (sqrt(sum(col("xx"))) * sqrt(sum(col("yy")))).alias("dem"),
        count(col("xy")).alias("pairsNum")
      )

    import spark.implicits._
    val pairSimilarityCalculated = pairSimilarity
      .withColumn("similarity",
        when(col("dem") =!= 0, col("num")/col("dem"))
          .otherwise(null)
      )
    pairSimilarityCalculated.select("ISBN1", "ISBN2",
      "similarity", "pairsNum").as[PairSimilarity]
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("BookRecommendations")
      .master("local[*]")
      .config("spark.local.dir", "memo")
      .getOrCreate()

      println("Loading data...\n")

    import spark.implicits._
    val ratingsRaw = spark.read
      .option("sep", ";")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/BX-Book-Ratings.csv")
      .as[Rating]

    val booksRaw = spark.read
      .option("sep", ";")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/BX_Books.csv")
      .as[Book]

    val usersRaw = spark.read
      .option("sep", ";")
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/BX-Users.csv")
      .as[User]

    println("Processing...\n")

    val ratingsFixed = ratingsRaw.filter($"BookRating" =!= 0)


    val trainTestSet = ratingsFixed.randomSplit(Array(0.1, 0.9))(1)
    val ratingPairs = trainTestSet.as("ra1")
      .join(ratingsRaw.as("ra2"),
        $"ra1.UserID" === $"ra2.UserID" && $"ra1.ISBN" < $"ra2.ISBN")
      .select($"ra1.ISBN".alias("ISBN1"),
      $"ra2.ISBN".alias("ISBN2"),
      $"ra1.BookRating".alias("Rating1"),
      $"ra2.BookRating".alias("Rating2"),
        $"ra1.UserID".alias("UserID")
    ).repartition(100).as[RatingPair]


    val PairSimilarities = computeCosineSimilarity(spark, ratingPairs).sort($"pairsNum".desc).cache()
    PairSimilarities.show()
    println(PairSimilarities.count().toInt)

    spark.stop()
  }
}
