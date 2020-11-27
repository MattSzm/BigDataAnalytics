package mattszm.kagglewithspark

import org.apache.spark.sql._
import org.apache.log4j._


object VideoGameSales {

  case class Sale(Rank: Int, Name: String, Platform: String,
                  Year: Int, Genre: String, Publisher: String,
                  NA_Sales: Float, EU_Sales: Float,
                  JP_Sales: Float, Other_Sales: Float)

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)

  }
}
