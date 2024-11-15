# Wprowadzenie
Dlaczego Transfer learning?
Poniewaz wymaga on znacznie mniej mocy obliczeniowej w porównaniu do konwencjonalnego własnoręcznie zrobionego CNN + DNN, umożliwiajac znacznie lepszą ekstrakcję cech.

Automatyczne generowanie opisów stanowi połączenie dwóch kluczowych dziedzin sztucznej inteligencji: widzenia komputerowego i przetwarzania języka naturalnego. Dotychczasowe podejścia dzielą się na odgórne, rozpoczynające się od ogólnej interpretacji obrazu, którą następnie przekształca się w słowa, oraz oddolne, zaczynające od doboru słów opisujących różne aspekty obrazu, które są następnie ze sobą łączone.

## first
./flickr8k/archive/first

Pierwsze badanie prostej sieci głębokiej o modelu z rysunku densenet_DNN. Podawane są na wejście cechy z obrazów uzyskane za pomocą sieci Densenet (2k cech).\
3h
## second
Drugie badania na takim samym modelu jak w pierwszym przypadku tylko badanie na cechach VGG16 (4k cech).\
3h
## third
Lekkie przemienienie sieci pierwszej i badania na cechach Densenet (2k cech).\
4h
## fourth
chęć stworzenia sieci, która pozwoli na wykorzystanie bazy danych oraz 4 tys cech z sieci VGG

Stworzenie wlasniego modelu, dluzszego ale płytkiego\
7h
# fifth
Tak samo jak w poprzednim, ale zwiększenie długości i zwiększenie szerokości\
22h

# sixth
Tak samo jak w fifth tylko zbadania tezy o zmniejszeniu dropoutu, ze względu na wyniki badania loss.\
14h

# Opis działania
Przetestowanie sieci z Internetu w celu obrania punktu startowego, oraz sprawdzenia poprawności działania programu. Składał się on z metody transfer learning, wybranej na podstawie artykułów naukowych oraz badań ze względu na jakość dostarczanych cech oraz zmniejszenia czasu działania programu jak i wymogów mocy obliczeniowej. Badanie **first** zostało przeprowazdone uzywając właśnie tej metody oraz wyzej opisanej sieci z podawaniem na wejście cech z modelu **Densenet**, wyłapującym 2048 cech obrazu. Następnie, aby zbadać czy działanie modelu poprawi/pogorszy się przy podaniu 4096 cech podano na wejście cechy z modelu **VGG16**. Wyniki opisane zostały w plikach loss w postaci obrazu. Kolejnym krokiem było sprawdzenie wpływu zmiany sieci na ilość epok oraz sprawności predykcji **third** jest programem, lekko zmienionej sieci w celu sprawdzenia czy nastąpi poprawa, głównie zostały zmienione warstwy sieci. **fourth** jest siecią zbudowaną na podstawie artykułów znalezionych na internecie, czytając je sugerowałem się opisami co każdy z modli robi i jak każda z warstw wpływa na ogólną wydajność. Samo dobranie wielkości sieci było eksperymentalne, tak aby jej uczenie nie zajęło dużo czasu. Model powstał z zamiarem stworzenia solidnego szkieletu do rozwiazywania naszego problemu. Ostania z nich **fifth** dobrana została podobnie jak **fourth** tylko ze znacznie zwiększoną wiedzą o warstwach w problemie opisywania obrazów. Siec nie jest aż taka płytka jak w przypadku wcześniejszych problemów. Została znacznie zwiększona i rozciągnięta. Do modelu dodane zostały także warstwy nie występujące wcześniej w żadnym w modeli. **sixth** w porównaniui z **fifth** posiada taki sam model, jedynymi czynnikami zmienionymi są dropouty które z wartości: 0.5, 0.7, 0.5, zmienione zostaly na 0.2, 0.3, 0.2

# Tabela
| Model              | Parameters| Input |
| :---------------- | :------: | ----: |
| first        |   4.316.709   | Dense (2k) |
| second           |   4.316.709   | VGG16 (4k) |
| third    |  2.708.203   | Dense (2k) |
| fourth |  6.388.005   | VGG16 (4k) |
| fifth | 13.087.781 | VGG16 (4K) |
| sixth | 13.087.781 | VGG16 (4K) |

# Wykresy
## categorical_crossentropy
W modelu generacji podpisów obrazów, który masz w kodzie, celem jest przewidzenie następnego słowa w sekwencji na podstawie poprzednich słów oraz cech obrazu. Używa się kategorii krzyżowej (categorical_crossentropy) jako funkcji strat, ponieważ:

    Problem klasyfikacji słów: Model próbuje przewidzieć każde kolejne słowo w sekwencji spośród wszystkich możliwych słów w słowniku. Jest to więc zadanie wieloklasowej klasyfikacji, gdzie każda kategoria reprezentuje jedno słowo.

    Reprezentacja wyjścia jako klasy: Wartości przewidywane przez model są reprezentowane jako prawdopodobieństwa dla każdego słowa w słowniku (na końcowej warstwie softmax). categorical_crossentropy jest stosowana do porównania przewidywanej rozkładu prawdopodobieństw z rzeczywistą kategorią (słowem) w postaci zakodowanego jedynką wektora (one-hot), gdzie 1 znajduje się tylko przy prawdziwym słowie, a 0 przy pozostałych.

    Skuteczność w zadaniach NLP: categorical_crossentropy jest powszechnie stosowana w zadaniach NLP, takich jak tłumaczenie czy generowanie sekwencji (w tym podpisów do obrazów), ponieważ umożliwia efektywne uczenie modelu, by koncentrował się na prawdopodobnych słowach dla danego kontekstu sekwencji.

## fifth
**Funkcja strat (loss function)** – W kodzie korzystamy z funkcji categorical_crossentropy, która jest używana głównie w zadaniach klasyfikacyjnych i daje wartości logarytmiczne. Wartości wyjściowe tej funkcji mogą być stosunkowo wysokie, zwłaszcza na początku treningu, szczególnie przy dużym słowniku (liczba możliwych słów) i początkowo wysokim learning rate. Dla problemu klasyfikacji słów w zdaniu model oblicza stratę dla każdego słowa przewidywanego w sekwencji, co może prowadzić do wyższej wartości loss, ponieważ błędy z różnych słów mogą się kumulować.

**Rozmiar słownika (vocab_size)** – Duży słownik oznacza więcej możliwości do przewidzenia, co zwiększa trudność zadania i może prowadzić do wyższego loss na początku treningu. W kodzie vocab_size jest obliczany dynamicznie na podstawie liczby unikalnych słów w korpusie tekstu.

**Losowe inicjalizowanie wag modelu** – Na początku treningu wagi modelu są losowe, co może powodować większą stratę (loss). Z czasem, gdy model uczy się i dostosowuje wagi, wartość ta powinna zacząć spadać.

**Dostosowywanie learning rate** – W kodzie dodano ReduceLROnPlateau, aby dostosowywać learning rate na podstawie wyników walidacyjnych, co powinno pomóc w obniżeniu loss z czasem. Na początku jednak, zanim mechanizm zadziała, loss może być wysoki.

**One-hot encoding** – Ponieważ klasy wyjściowe są zakodowane za pomocą funkcji to_categorical, każda przewidywana klasa jest wektorem wielkości vocab_size. Jeśli wartość prawdziwego słowa jest niska, to błąd na każdej iteracji treningowej (na każdym tokenie) może znacząco zwiększać stratę.

## third
Obraz w archiwum trzecim wykres strat modelu (loss) w funkcji liczby epok treningowych, po zmniejszeniu złożoności modelu sześciokrotnie w porównaniu z piątym ostatnim modelem. Spójrzmy na ten wykres:

    - Strata treningowa (Training Loss) - Niebieska linia wciąż spada, choć wolniej niż na wykresie z poprzednim modelem. To sugeruje, że mniejszy model nadal uczy się, ale już wolniej zmniejsza swoją stratę, co może wskazywać na ograniczenie jego zdolności do pełnego odwzorowania skomplikowanych relacji w danych.

    - Strata walidacyjna (Validation Loss) - Pomarańczowa linia pozostaje stabilna lub waha się na określonym poziomie, co może sugerować mniejsze przeuczenie w porównaniu do poprzedniego wykresu, ale też brak poprawy w zdolności generalizacji.

Wnioski:

Zmniejszenie modelu wydaje się pomóc w stabilizacji straty walidacyjnej, ale może być zbyt radykalne, skoro strata walidacyjna przestała spadać lub osiągnęła plateau. Być może warto spróbować z modelem o pośredniej złożoności lub z dostrojeniem hiperparametrów, jak:

    Dodanie dropout na poziomie umiarkowanym, eksperymentowanie z mniejszym początkowym współczynnikiem uczenia, aby model uczył się wolniej, ale być może bardziej efektywnie.

## sixth 
Po analizie obrazu piątego wniosek: zmniejszenie parametru dropout moze wspomóc uczenie się sieci. Analiza: znacznie poprawilo loss treningowy ale zmniejszylo ilosc epok, nie zwracając uwagi na walidacje tylko na trening, model znacznie lepiej radzi sobie będąc większą strukturą pod względem szerokości i głębokości. Walidacja moze byc słaba ze względu na ilość epok. Wniosek: istnieje większa szansa na poprawę gdy dodamy duzo wiecej eopk (małe prawdopodobieństwo), znacze większe prawdopobieństwo zwracając także uwagę na tokenizacje oraz obrazy, baza danych posiada znacznie wiecej pewnych opisów takich jak "pies biegający po trawie" przez co klasteryzacja danych skupia sie w pewnych miejscach, przez co model moze tracić na ogólności.

# Warstwy
- embedding -  Warstwa embedding przekształca tokeny (czyli identyfikatory słów) w wektory liczbowe, które oddają ich semantyczne znaczenie. Na przykład, słowa o zbliżonym znaczeniu otrzymują podobne wektory, co pozwala modelowi lepiej uchwycić wzajemne relacje między słowami i ich znaczenie. W modelach generujących opisy obrazów istotne jest rozpoznawanie semantycznych powiązań między słowami, co wpływa na spójność i logikę generowanych zdań. Warstwa embedding odwzorowuje każdy element wejściowy (słowo) na wektor o niższym wymiarze, który przechowuje informacje o relacjach między różnymi elementami. Dzięki temu słowa powiązane znaczeniowo otrzymują podobne reprezentacje wektorowe. Embedding stanowi znaczący poizom przetwarzania w języku naturalnym, umożliwiając sieciom neuronowym skuteczniejsze rozpoznawanie kontekstu i znaczenia.\
**Zalety**: Pozwala na reprezentowanie słów i innych elementów w sposób numeryczny, co ułatwia ich dalsze przetwarzanie. Przyspiesza trening i pozwala na lepsze zrozumienie danych sekwencyjnych.\
**Wady**: Wymaga odpowiedniego doboru wymiaru przestrzeni osadzania. Źle wytrenowane embeddings mogą pogorszyć wyniki modelu.

- batch_normalization - Normalizacja wsadowa stabilizuje proces uczenia się przez zmniejszenie „przesunięcia wewnętrznego” (internal covariate shift). Oznacza to, że dane wejściowe są bardziej stabilne w czasie uczenia, co ułatwia i przyspiesza trening. Batch normalization zwiększa stabilność modelu i zmniejsza ryzyko przeuczenia. Dzięki temu model koncentruje się na istotnych informacjach w cechach obrazu i łatwiej osiąga dobrą wydajność na nowych danych. Normalizacja wsadowa oblicza średnią i odchylenie standardowe dla każdego wsadu danych i przeskalowuje oraz przesuwa dane, aby miały ustandaryzowaną rozkład zbliżony do normalnego (średnia zero, odchylenie standardowe jeden). Batch Normalization jest szeroko stosowany w sieciach konwolucyjnych (CNN) oraz w głębokich sieciach neuronowych, aby przyspieszyć uczenie i poprawić stabilność.\
**Zalety**: Przyspiesza trening i zapobiega przeuczeniu. Może także pomóc w uzyskaniu lepszej dokładności modelu.\
**Wady**: Dodaje dodatkowe operacje obliczeniowe i czasami może prowadzić do nadmiernej zależności od danych treningowych.

- bidirectional(LSTM) - Dwukierunkowy LSTM ma dwie części – jedną analizującą sekwencję od początku do końca, a drugą od końca do początku. Dzięki temu model może zrozumieć zarówno wcześniejszy, jak i późniejszy kontekst każdego słowa. Model zyskuje pełniejszą perspektywę na sekwencję słów, co umożliwia mu tworzenie bardziej spójnych i naturalnych zdań. W kontekście generowania opisów obrazów model może lepiej zrozumieć relacje między słowami. Warstwa bidirectional łączy dwie oddzielne sieci RNN, które przetwarzają sekwencję w przeciwnych kierunkach. W ten sposób model ma dostęp zarówno do poprzedniego, jak i przyszłego kontekstu dla każdego elementu sekwencji.\
**Zalety**: Pozwala modelowi na bardziej dokładne rozumienie sekwencji dzięki dostępowi do pełnego kontekstu.\
**Wady**: Dwukierunkowa struktura zwiększa złożoność obliczeniową i pamięciową modelu, co może wydłużyć czas treningu.

- attention - Warstwa ta analizuje, które słowa w sekwencji są najważniejsze przy przewidywaniu kolejnego słowa. Mechanizm ten przypisuje wagi do różnych słów, umożliwiając modelowi elastyczne przetwarzanie informacji z różnych części sekwencji. Uwaga (attention) poprawia zdolność modelu do generowania kontekstowo odpowiednich słów, co jest kluczowe dla płynności i poprawności gramatycznej opisów. Mechanizm uwagi pozwala modelowi skupić się na istotnych częściach danych wejściowych podczas przetwarzania sekwencji. Attention umożliwia modelowi wybór, które elementy sekwencji są najważniejsze dla danego zadania. Mechanizm uwagi przypisuje różne wagi (atencje) do elementów sekwencji wejściowej. Model oblicza wagi na podstawie kontekstu i następnie wykorzystuje je do zsumowania istotnych informacji. W rezultacie model uczy się, które fragmenty wejścia są najważniejsze dla przewidywania wyjścia.\
**Zalety**: Umożliwia lepsze zrozumienie kontekstu i wyodrębnianie istotnych informacji, co poprawia jakość wyników.\
**Wady**: Attention może być kosztowny obliczeniowo, szczególnie dla długich sekwencji.

- concatenate - Połączenie tych trzech komponentów pozwala modelowi na utworzenie jednolitej reprezentacji, która łączy informacje z obrazu i sekwencji słów. Dzięki temu model ma pełniejszy obraz kontekstu – łączy zarówno informacje wizualne, jak i tekstowe, co pozwala na bardziej spójne generowanie opisów obrazów.

- LSTM - Warstwa LSTM analizuje złożoną reprezentację, aby wyodrębnić najważniejsze informacje z kombinacji obrazu i sekwencji słów. Wektor wyjściowy z LSTM zawiera kontekstowo najważniejsze informacje, które model wykorzystuje do przewidywania następnego słowa w opisie obrazu. LSTM to specjalny typ sieci rekurencyjnej (RNN), zaprojektowany do przetwarzania sekwencji danych, takich jak tekst, dźwięk lub sekwencje czasowe. LSTM radzi sobie z tzw. problemem "zanikającego gradientu", który utrudnia tradycyjnym RNN efektywne trenowanie na długich sekwencjach. LSTM ma specjalne komórki pamięci, które pozwalają mu "zapamiętywać" istotne informacje przez długi czas. Każdy neuron LSTM zawiera trzy główne bramki:
    - Bramka wejściowa: decyduje, jakie nowe informacje powinny zostać zapisane w komórce.
    - Bramka zapominania: decyduje, które informacje z komórki pamięci powinny zostać usunięte.
    - Bramka wyjściowa: kontroluje, które informacje z komórki pamięci zostaną użyte jako wyjście.\
**Zalety**: Doskonale nadaje się do przetwarzania długich sekwencji danych, zachowując informacje kontekstowe.\
**Wady**: Jest stosunkowo skomplikowana i kosztowna obliczeniowo, co sprawia, że jest trudniejsza do trenowania niż tradycyjne RNN.

- Dropout - wyłącza część neuronów w czasie treningu, co redukuje przeuczenie. Regularizacja Dropout przeciwdziała nadmiernemu dopasowaniu modelu do danych treningowych, poprawiając jego zdolność generalizacji. Przy każdej iteracji treningu część neuronów zostaje wyłączona (zazwyczaj procentowo, np. 50%). Dzięki temu sieć neuronowa nie jest w stanie polegać zbyt mocno na pojedynczych neuronach i musi nauczyć się bardziej ogólnych cech. Dropout jest powszechnie stosowany w głębokich sieciach neuronowych, aby zmniejszyć ryzyko przeuczenia, zwłaszcza gdy mamy ograniczoną ilość danych treningowych.\
**Zalety**: Dropout skutecznie zapobiega przeuczeniu i poprawia ogólną wydajność modelu na danych testowych.\
**Wady**: Może spowolnić proces treningu, ponieważ losowe wyłączanie neuronów powoduje zmiany w obliczeniach podczas każdej iteracji.

- dense - Kolejny poziom nieliniowości pozwala modelowi lepiej nauczyć się złożonych interakcji między cechami wizualnymi a tekstowymi. Dzięki temu model uzyskuje bardziej precyzyjne wyjście, co umożliwia dokładniejsze przewidywanie kolejnego słowa. Warstwa ta wykonuje operację macierzową na swoich wejściach. Każdy neuron oblicza sumę ważoną wejść plus bias, a następnie przekazuje wynik przez funkcję aktywacji (np. ReLU, sigmoid, tanh), aby nadać sieci nieliniowość. Warstwa Dense jest używana do "uczenia się" abstrakcyjnych wzorców w danych. Na przykład w modelach klasyfikacyjnych ostatnia warstwa Dense z odpowiednią liczbą neuronów (równej liczbie klas) jest używana do przewidywania prawdopodobieństw dla każdej klasy.\
**Zalety**: Jest uniwersalna i prosta w implementacji. Dobrze sprawdza się w wielu typach zadań.\
**Wady**: Zajmuje dużo pamięci, gdy liczba neuronów jest duża, i może łatwo prowadzić do przeuczenia w przypadku skomplikowanych danych.

- GRU - W przeciwieństwie do LSTM, GRU ma tylko dwie bramki (update i reset gate) zamiast trzech. Jest mniej skomplikowana niż LSTM, ale wciąż potrafi przechowywać kontekst sekwencyjny przez dłuższy czas. GRU znajduje zastosowanie w przetwarzaniu języka naturalnego, analizie sekwencji czasowych i wszędzie tam, gdzie potrzebny jest kontekst.\
**Zalety**: Jest szybsza i prostsza niż LSTM, co czyni ją lepszą opcją w przypadku, gdy zależności czasowe są mniej skomplikowane.\
**Wady**: W niektórych zadaniach LSTM może być bardziej skuteczna, ponieważ ma bardziej zaawansowany mechanizm pamięciowy.

- Residual Layer - Mechanizm "skip connection" lub "shortcut" pozwala modelowi nauczyć się, jak modyfikować dane bez ryzyka, że wartości zanikną lub eksplodują w głębokiej sieci. Stosowane w bardzo głębokich sieciach neuronowych, aby poprawić gradient i stabilność sieci.\
**Zalety**: Pozwala budować bardzo głębokie sieci bez ryzyka zanikających gradientów.\
**Wady**: Może prowadzić do większej złożoności obliczeniowej.
