## Greedy Algorithm (Шуналт алгоритм) — Ерөнхий ойлголт

Шуналт (Greedy) алгоритм нь асуудлыг шийдэхдээ "аль одоогоор хамгийн ашигтай" гэж бодогдож байгаа алхмыг шат бүрт хийдэг.

Онцлог:
- Хэрэв тухайн шилдэг алхам нь ирээдүйд ч үргэлж шилдэг байх боломжтой бол уг алгоритм нь глобаль оптимумд хүрнэ.
- Энгийн, хурдан, хэрэглэхэд ойлгомжтой.
- Зарим тохиолдолд (зарим мөнгөн зоосны систем, нарийн онцлогтой бодлогууд) төгс (optimal) шийдэл бус байж болно.

Жишээ: Зоосоор мөнгө төлөх (Coin Change)

Зорилго: Өгөгдсөн зооснуудын бүрдлээс нийт дүнг хамгийн цөөн зоосоор төлөх.

Алгоритм: Том зоосноос эхлэн боломжтой тохиолдолд аль олонг авч болохыг авна.

```python
# GREEDY ALGORITHM – Coin Change Problem

def greedy_coin_change(coins, amount):
    """coins: зооснуудын жагсаалт (тоон утга)
    amount: төлөх дүн
    Буцаана: (зоос, тоо) хосуудын жагсаалт"""

    # Том-бага руу эрэмбэлэ
    coins = sorted(coins, reverse=True)

    result = []
    for coin in coins:
        count = amount // coin
        if count:
            result.append((coin, count))
            amount -= coin * count

    return result

# Турших
if __name__ == '__main__':
    coins = [25, 10, 5, 1]
    amount = 93
    change = greedy_coin_change(coins, amount)
    print("Greedy зоос:", change)  # [(25, 3), (10, 1), (5, 1), (1, 3)]
```

Тайлбар:
- Дээрх жишээнд Greedy-н шийдэл нь төгс (93 = 25*3 + 10*1 + 5*1 + 1*3).
- Гэхдээ бүх зоосны систем дээр Greedy үргэлж төгс биш (зарим тусгай зоосны бүрдэлд динамик болон бусад аргууд хэрэгтэй).

---

## Dijkstra Algorithm (Дейкстра) — Богино зам олох алгоритм

Dijkstra нь нэг эхлэл цэгээс бусад бүх зангилаа хүртэлх хамгийн богино замыг олдог алгоритм юм. Энэ нь Greedy логик ашигладаг — алхам бүр "одоо хамгийн ойр" зангилааг шийдэлд тогтоодог.

Хязгаарлалтууд:
- Эерэг жинтэй (non-negative) ирмэгийн жинтэй графт зөв ажиллана.
- Сөрөг жингтэй графын хувьд ажиллахгүй (үүнд Bellman-Ford хэрэгтэй).

```python
import heapq

# DIJKSTRA ALGORITHM

def dijkstra(graph, start):
    """graph: {node: {neighbor: weight, ...}, ...}
    start: эхлэх зангилаа
    Буцаана: start-с бусад зангилаа руу хамгийн богино зай-уудын dict"""

    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    pq = [(0, start)]  # (зай, зангилаа)

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # Хэрэв хураагдсан зай илүү урт бол алгасна
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances

# Турших
if __name__ == '__main__':
    graph = {
        'A': {'B':5, 'C':2, 'D':9},
        'B': {'A':5, 'D':1},
        'C': {'A':2, 'D':6},
        'D': {'A':9, 'B':1, 'C':6}
    }

    start_node = 'A'
    shortest_paths = dijkstra(graph, start_node)

    print(f"Dijkstra үр дүн ({start_node}-с):")
    for node, distance in shortest_paths.items():
        print(f"{start_node} → {node} = {distance}")

# Гаралт:
# A → A = 0
# A → B = 5
# A → C = 2
# A → D = 6
```

Алгоритмын санаа (алхмууд):
1. Бүх зангилааг эхэнд ∞ гэж тэмдэглэх, эхлэлийн зайг 0 тавина.
2. Priority queue-ээс хамгийн бага зайтай зангилааг авна.
3. Түүний хөршүүдийн зайг шинэчлэн, илүү богино зай олбол priority queue-д нэмнэ.
4. Бүх зангилаа шалгагдтал давталт үргэлжилнэ.

---

## Greedy vs Dijkstra — Ялгаа, холбоо

- Dijkstra бол Greedy зарчмаар ажилладаг тусгай алгоритм — алхам бүр хамгийн ойр зангилааг "фикс" хийнэ.
- Greedy — ерөнхий стратеги (жишээ: coin change, activity selection, Huffman coding). Зарим асуудалд хамгийн сайн шийдлийг өгдөг, заримд нь үгүй.
- Dijkstra-ийн хувьд: нэг эхлэлийн хамгийн богино замыг баталгаатай олно (граф эерэг жинтэй байх нөхцөлд).

Товч харьцуулалт:
- Зорилго: Greedy — ерөнхий; Dijkstra — граф дахь богино зам.
- Баталгаа: Greedy — заримдаа (problem-dependent) сайн; Dijkstra — зөв (non-negative weights).

---

## Хэрэглэх заавар / Дараагийн алхмууд

- Хэрэв та `greedy.md`-г сольж, сайжруулсан хувилбарыг ашиглахыг хүсвэл `greedy_improved.md` файлыг `greedy.md` дээр давхарлан хуулахад болно.
- Хүсвэл би таны хүссэн дагуу оригинал файлыг шууд өөрчлөх боломжтой.
- Өгөгдлийн нарийн нөхцөлүүдийн дагуу (жишээ: зооснуудын систем өөр байвал) Greedy-н хувилбар тусгүй болох шинэ жишээнүүд нэмэхийг санал болгож байна.

---

## Хураангуй

- Greedy нь энгийн, хурдацтай шийдэл олдог техниктэй.
- Dijkstra нь Greedy-гийн нэг тусгай хэрэглэл бөгөөд граф дээр хамгийн богино замыг олдог.
- `greedy_improved.md`-д тайлбар, код, харьцуулалт цэгцтэй байна — та хүсвэл би файлыг оригинал дээр нэхэмжлэх (replace) маягаар өөрчилж өгнө.

---

## Нэмэлт: Илүү дэлгэрэнгүй жишээнүүд (Алхам алхмаар)

Доорх хэсгүүд нь таны хүссэн "жишээ болон бодлого хамт, тайлбарласан" маягтыг хангана: Coin Change-д Greedy хэрхэн ажиллаж байгааг бодитоор үзүүлж, нэг counterexample-ыг DP-ээр баталгаажуулна. Мөн Dijkstra-д замыг сэргээж, алхам бүр тайлбарлана.

### 1) Coin Change — Greedy ажилж буй жишээ

Зоос: [25, 10, 5, 1], Дүн = 93

Алхамууд:
- Том зоос 25: 93 // 25 = 3 → авна. Үлдэгдэл 93 - 75 = 18
- 10: 18 // 10 = 1 → авна. Үлдэгдэл 8
- 5: 8 // 5 = 1 → авна. Үлдэгдэл 3
- 1: 3 // 1 = 3 → авна. Үлдэгдэл 0

Итгэлтэй шийдэл: [(25,3),(10,1),(5,1),(1,3)] → нийт 8 зоос.

Энэ жишээнд Greedy нь оптималь.

### 2) Coin Change — Greedy-н буруу шийдэл (Counterexample)

Зоос: [1, 3, 4], Дүн = 6

Greedy-н алхмууд:
- Том зоос 4-г авна (1 ширхэг). Үлдэгдэл 2.
- Дараа нь 1-үүдийг хоёр (1+1) авна.

Greedy шийдэл: 4 + 1 + 1 → 3 зоос.
Гэтэл оптималь шийдэл бол 3 + 3 → 2 зоос.

Доорх код нь Greedy-г ажиллуулсан үр дүнг харуулж, дараа нь динамик програмчлал ашиглан хамгийн цөөн зоосыг олоод харьцуулна.

```python
# Greedy vs DP coin change example
from collections import Counter

def greedy_coin_change(coins, amount):
    coins = sorted(coins, reverse=True)
    result = []
    for coin in coins:
        count = amount // coin
        if count:
            result.append((coin, count))
            amount -= coin * count
    return result

def dp_min_coins(coins, amount):
    # dp[i] = хамгийн цөөн зоос i-ийг төлөхөд
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    parent = [-1] * (amount + 1)
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1
                parent[i] = c
    if dp[amount] == float('inf'):
        return None, None
    res = []
    a = amount
    while a > 0:
        res.append(parent[a])
        a -= parent[a]
    return dp[amount], Counter(res)

if __name__ == '__main__':
    coins = [1, 3, 4]
    amount = 6
    print('Greedy result:', greedy_coin_change(coins, amount))
    dp_count, dp_comb = dp_min_coins(coins, amount)
    print('DP optimal min coins:', dp_count, 'combination:', dict(dp_comb))

# Expected output:
# Greedy result: [(4, 1), (1, 2)]
# DP optimal min coins: 2 combination: {3: 2}
```

#### Тайлбар:
- Энэ нь Greedy-гийн "одоогоор хамгийн сайн" шийдэл нь зарим системд глобал оптимум биш байж болохыг тод харуулна.
- Ийм тохиолдолд DP (эсвэл бусад аргыг) ашиглан баталгаажуулах шаардлагатай.

### 3) Dijkstra — Зам сэргээн босгох ба алхам алхмаар тайлбар

Доорх Dijkstra хэрэгжүүлэлт нь зайг тооцоолно, мөн `parent` массив ашиглан эхнээс зорьсон зангилаа хүртэлх замыг сэргээнэ.

```python
import heapq

def dijkstra_with_path(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    parent = {node: None for node in graph}

    pq = [(0, start)]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]:
            continue
        for nei, w in graph[node].items():
            nd = dist + w
            if nd < distances[nei]:
                distances[nei] = nd
                parent[nei] = node
                heapq.heappush(pq, (nd, nei))
    return distances, parent

def reconstruct_path(parent, start, goal):
    if parent[goal] is None and start != goal:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

if __name__ == '__main__':
    graph = {
        'A': {'B':5, 'C':2, 'D':9},
        'B': {'A':5, 'D':1},
        'C': {'A':2, 'D':6},
        'D': {'A':9, 'B':1, 'C':6}
    }
    distances, parent = dijkstra_with_path(graph, 'A')
    print('Distances:', distances)
    print('Path A -> D:', reconstruct_path(parent, 'A', 'D'))

# Алхам алхмаар (хураангуй):
# 1) Эхэнд A=0, бусад = inf
# 2) PQ-с C (2) олдож, C-гийн хөршүүдээр дамжуулан D-г 8 болгоно (9-с бага)
# 3) Дараа PQ-с B (5) ирж, B-гээс D = 6 болгоно (8-с бага)
# 4) Эцэст нь D зай 6-т тогтнох ба сэргээн босгоход A->B->D гарна.
```

---

## Дүгнэлт

- Жишээгээр харахад Greedy нь ихэвчлэн хурдан, ойлгомжтой шийдэл өгдөг, гэхдээ бүх системд оптималь биш.
- Иймд coin-change зэрэг зарим бодлогуудыг баталгаажуулахын тулд DP эсвэл бусад техникийг ашиглах шаардлагатай.
- Dijkstra нь Greedy ойлголт дээр суурилсан боловч графын богино замыг баталгаатай олох тусгай арга бөгөөд зам сэргээн босгох боломжтой.

Хэрвээ та хүсвэл би `greedy.md` файлыг шууд орлуулах (replace) байдлаар шинэчилж өгье, эсвэл эдгээр жишээнүүдийг тусдаа `examples/` фолдерт байрлуулж, богино тест (unit tests) нэмэх боломжтой.
