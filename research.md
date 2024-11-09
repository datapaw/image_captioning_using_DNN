# Wprowadzenie
Dlaczego Transfer learning?
Poniewaz wymaga on znacznie mniej mocy obliczeniowej w porównaniu do konwencjonalnego własnoręcznie zrobionego CNN + DNN, umożliwiajac znacznie lepszą ekstrakcję cech.

## first
./flickr8k/archive/first

Pierwsze badanie prostej sieci głębokiej o modelu z rysunku densenet_DNN. Podawane są na wejście cechy z obrazów uzyskane za pomocą sieci Densenet (2k cech).
3h
## second
Drugie badania na takim samym modelu jak w pierwszym przypadku tylko badanie na cechach VGG16 (4k cech).
3h
## third
Lekkie przemienienie sieci pierwszej i badania na cechach Densenet (2k cech).
4h
## fourth
chęć stworzenia sieci, która pozwoli na wykorzystanie bazy danych oraz 4 tys cech z sieci VGG

Stworzenie wlasniego modelu, dluzszego ale płytkiego
7h
# fifth
Tak samo jak w poprzednim, ale zwiększenie długości i zwiększenie szerokości


# Opis działania
Przetestowanie sieci z Internetu w celu obrania punktu startowego, oraz sprawdzenia poprawności działania programu. Składał się on z metody transfer learning, wybranej na podstawie artykułów naukowych oraz badań ze względu na jakość dostarczanych cech oraz zmniejszenia czasu działania programu jak i wymogów mocy obliczeniowej. Badanie **first** zostało przeprowazdone uzywając właśnie tej metody oraz wyzej opisanej sieci z podawaniem na wejście cech z modelu **Densenet**, wyłapującym 2048 cech obrazu. Następnie, aby zbadać czy działanie modelu poprawi/pogorszy się przy podaniu 4096 cech podano na wejście cechy z modelu **VGG16**. Wyniki opisane zostały w plikach loss w postaci obrazu. Kolejnym krokiem było sprawdzenie wpływu zmiany sieci na ilość epok oraz sprawności predykcji **third** jest programem, lekko zmienionej sieci w celu sprawdzenia czy nastąpi poprawa, głównie zostały zmienione warstwy sieci. **fourth** jest siecią zbudowaną na podstawie artykułów znalezionych na internecie, czytając je sugerowałem się opisami co każdy z modli robi i jak każda z warstw wpływa na ogólną wydajność. Samo dobranie wielkości sieci było eksperymentalne, tak aby jej uczenie nie zajęło dużo czasu. Model powstał z zamiarem stworzenia solidnego szkieletu do rozwiazywania naszego problemu. Ostania z nich **fifth** dobrana została podobnie jak **fourth** tylko ze znacznie zwiększoną wiedzą o warstwach w problemie opisywania obrazów. Siec nie jest aż taka płytka jak w przypadku wcześniejszych problemów. Została znacznie zwiększona i rozciągnięta. Do modelu dodane zostały także warstwy nie występujące wcześniej w żadnym w modeli.

# Tabela
| Model              | Parameters| Input |
| :---------------- | :------: | ----: |
| first        |   4.316.709   | Dense (2k) |
| second           |   4.316.709   | VGG16 (4k) |
| third    |  2.708.203   | Dense (2k) |
| fourth |  6.388.005   | VGG16 (4k) |
| fifth | 13.087.781 | VGG16 (4K) |

# Wykresy


# Warstwy

