package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions._
import scala.util.Random.nextInt


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

  def createPairSimilarities(spark: SparkSession, data: Dataset[Rating],
                             lowLimit: Int): Dataset[PairSimilarity] = {
    import spark.implicits._

    val dataFixed = data.filter($"BookRating" >= lowLimit)
//    val trainTestSet = dataFixed.randomSplit(Array(0.5, 0.5))(0)
    val ratingPairs = dataFixed.as("ra1")
      .join(dataFixed.as("ra2"),
        $"ra1.UserID" === $"ra2.UserID" && $"ra1.ISBN" < $"ra2.ISBN")
      .select($"ra1.ISBN".alias("ISBN1"),
        $"ra2.ISBN".alias("ISBN2"),
        $"ra1.BookRating".alias("Rating1"),
        $"ra2.BookRating".alias("Rating2"),
        $"ra1.UserID".alias("UserID")
      ).repartition(200).as[RatingPair]

    computeCosineSimilarity(spark, ratingPairs)
  }

  def computeCosineSimilarity(spark: SparkSession, data: Dataset[RatingPair])
  : Dataset[PairSimilarity] = {
    val pairScores = data
      .withColumn("xx", col("Rating1") * col("Rating1"))
      .withColumn("yy", col("Rating2") * col("Rating2"))
      .withColumn("xy", col("Rating1") * col("Rating2"))
    val pairSimilarity = pairScores.groupBy("ISBN1", "ISBN2")
      .agg(
        sum(col("xy")).alias("num"),
        (sqrt(sum(col("xx"))) * sqrt(sum(col("yy"))))
          .alias("dem"),
        count(col("xy")).alias("pairsNum")
      )

    import spark.implicits._
    val pairSimilarityCalculated = pairSimilarity
      .withColumn("similarity",
        when(col("dem") =!= 0, col("num")/col("dem"))
          .otherwise(null)
      )
      .withColumn("similarity", round($"similarity", 4))
    pairSimilarityCalculated.select("ISBN1", "ISBN2",
      "similarity", "pairsNum").as[PairSimilarity]
  }

  def getBook(booksData: Dataset[Book], idBook: String): Book = {
    val foundBook = booksData.filter(
      col("ISBN") === idBook).collect()
    foundBook(0)
  }

  def displayResults(recommendationsData: Array[PairSimilarity],
                     booksData: Dataset[Book], idBook: String): Unit = {
    recommendationsData.foreach(singleRes => {
      var similarID = singleRes.ISBN1
      if (similarID == idBook) similarID = singleRes.ISBN2
      val foundBook = getBook(booksData, similarID)

      println(s"\'${foundBook.BookTitle}\', written by ${foundBook.BookAuthor}," +
        s" published by ${foundBook.Publisher} in ${foundBook.YearOfPublication} " +
        s"with similarity ratio=${singleRes.similarity}/1 " +
        s"and number of pairs=${singleRes.pairsNum}")
    })
  }

  def makeBetterDistribution(spark: SparkSession, data: Dataset[Rating]): Dataset[Rating] = {
    import spark.implicits._

    val fixedData = data.withColumn("BookRating",
      when(col("BookRating") === 0, nextInt(10)+1)
      .otherwise(col("BookRating")))
    fixedData.as[Rating]
  }

  def filterRatingsByAge(spark: SparkSession, ratingsData: Dataset[Rating],
                         usersData: Dataset[User], pickedRangeID: Int): Dataset[Rating] = {
    import spark.implicits._

    val ratingWithUsers = ratingsData.as("rF").join(usersData.as("uS"),
      $"rF.UserID" === $"uS.UserID")

    val ranges = List(List(0,20), List(20,40), List(40, 200))
    val ratingWithUsersFiltered = ratingWithUsers.filter(
      ($"Age" > ranges(pickedRangeID).head && $"Age" <= ranges(pickedRangeID).last)
        || $"Age" === "NULL")
    ratingWithUsersFiltered.select("rf.UserID", "ISBN", "BookRating").as[Rating]
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

    var idAgeRange: Int = 2
    if (args.length > 3) idAgeRange = args(3).toInt
    val ratingsFixed = makeBetterDistribution(spark, data=ratingsRaw)

    val ratingsProcessed: Dataset[Rating] = idAgeRange match {
      case ageRange: Int if 0 < ageRange && ageRange < 4 =>
        filterRatingsByAge(spark, ratingsData=ratingsFixed,
          usersData=usersRaw, pickedRangeID=ageRange-1)
      case _ => ratingsFixed
    }
    val pairSimilarities = createPairSimilarities(spark,
      data=ratingsProcessed, lowLimit=4).cache()


    var idBook: String = "0439139597"
    var similarityThreshold: Double = 0.98
    var occurrencesThreshold: Int = 10
    if (args.length > 0) idBook = args(0)
    if (args.length > 1) similarityThreshold = args(1).toDouble
    if (args.length > 2) occurrencesThreshold = args(2).toInt

    val recommendationResults = pairSimilarities.filter(
      (col("ISBN1") === idBook || col("ISBN2") === idBook) &&
      col("similarity") >= similarityThreshold &&
        col("pairsNum") >= occurrencesThreshold
    )
    val recommendationResultsSorted = recommendationResults
      .sort($"similarity".desc).collect()

    val foundBook = getBook(booksRaw, idBook)
    print(s"Book recommendations for: \'${foundBook.BookTitle}\' " +
      s"written by ${foundBook.BookAuthor}\n\n")
    displayResults(recommendationsData=recommendationResultsSorted,
      booksData=booksRaw, idBook=idBook)

    spark.stop()
  }
}
