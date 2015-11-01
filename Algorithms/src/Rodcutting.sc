def naive(prices: Array[Int], length: Int): Int = {
  if (length == 0) 0
  else {
    (for (l <- (1 to length)) yield prices(l - 1) + naive(prices, length - l)).toList.sortWith((a, b) => a > b).head
  }
}

//naive(Array(1,5,8,9,10,17,17,20,24,30,32,36,38,41,43,46,47,40,50,53,51,55,60,61,62,63,64,65,66,66,67,68), 26)

def memoizedCutRod(p: Array[Int], n: Int): Int = {
  val memo = Array.fill[Int](n+1)(-1)

  def memoizedCutRodAux(n: Int): Int = {
    if (memo(n) >= 0) memo(n)
    else if (n == 0) 0
    else {
      val q = (1 to n).map(i => p(i - 1) + memoizedCutRodAux(n - i)).max
      memo(n) = q
      q
    }
  }

  memoizedCutRodAux(n)
}

memoizedCutRod(Array(1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 32, 36, 38, 41, 43, 46, 47, 40, 50, 53, 51, 55, 60, 61, 62, 63, 64, 65, 66, 66, 67, 68, 70, 71, 74), 26)

def bottomUpCutRod(p: Array[Int], n: Int): Int = {
  val r = (0 +: p).foldLeft(Array[Int]()) {
    case (Array(), i) => Array[Int](i)
    case (arr, i) => arr :+ (for (l <- arr.indices) yield p(l) + arr(arr.length - l - 1)).max
  }
  r(n)
}

bottomUpCutRod(Array(1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 32, 36, 38, 41, 43, 46, 47, 40, 50, 53, 51, 55, 60, 61, 62, 63, 64, 65, 66, 66, 67, 68, 70, 71, 74), 26)