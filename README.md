# Слияние Krea2 SVD LoRA

Объедините несколько файлов LoRA в один и переупакуйте результат в единый ранг с помощью SVD. Работает полностью на процессоре, никогда не касается исходных файлов и оснащен небольшим графическим интерфейсом и режимом командной строки.

Типичное использование: у вас есть три Krea2 LoRA (стиль, персонаж, деталь LoRA) и вы хотите один `.сейфтензоры` файл с фиксированными коэффициентами, вместо того, чтобы складывать их во время генерации.

## Особенности

- **Взвешенное слияние** любого количества LoRA, с отдельным коэффициентом для каждого (допускаются отрицательные значения).
- **Единый выходной ранг** через QR + малоядерный SVD. Только маленький `кхк` матрица проходит через SVD, поэтому она быстрая даже для больших слоев.
- **Только процессор.** Графический процессор или видеопамять не требуются.
- **Управление потоками ЦП:** выберите 1–20 внутриоперационных потоков PyTorch в CLI или GUI; по умолчанию — 6. Большее количество потоков не приводит к автоматическому повышению скорости, поэтому протестируйте целевой процессор.
- **Запустить диагностику:** каждый запуск сообщает время настенных часов и начало/конец/пик выборочного процесса RSS. RSS — это память процесса, а не общая оперативная память системы; сэмплер длительностью 50 мс может пропускать очень короткие пики.
- **Предполетные проверки** при добавлении файла: структура SafeTensors, формы A/B, именование ключей, совместимость размеров с другими выбранными LoRA. Читаются только заголовки, так что это мгновенно.
- **Безопасно по конструкции:**
  - входные данные доступны только для чтения и никогда не могут быть выбраны в качестве выходных данных;
  - неизвестные тензоры останавливают пробег, а не молча отбрасываются;
  - результат записывается во временный файл, снова открывается и проверяется на форму и NaN/Inf, и только затем атомарно перемещается на место;
  - режим пробного запуска проверяет все без записи.
- **Сообщает об ошибке усечения** (средняя / максимальная относительная ошибка Фробениуса на слой), чтобы вы могли судить, достаточно ли высок целевой ранг.
- **Метаданные пользователя Forge Neo в формате JSON:** считывает каждый вход коляски LoRA `.джонсон`, комбайны `текст активации` и `негативный текст` фразы (дедупликация без учета регистра), доступны средние значения `предпочтительный вес` значения и записывает совместимый с Forge JSON из шести полей рядом с объединенным выводом. Отсутствующие входные файлы JSON пропускаются. Существующий выходной JSON защищен, если `--перезаписать` включено.
- **Современный графический интерфейс** построенный на [ttkbootstrap](https://github.com/israel-dryer/ttkbootstrap) с переключаемыми темами, журналом в реальном времени и индикатором прогресса. Возвращается на равнину `ткинтер.ттк` если ttkbootstrap не установлен.
  
<имг ширина="900" высота="997" альт="Экран" источник="https://github.com/user-attachments/assets/6782324e-df4e-4ca2-8863-6bacc9e2ea5f" />

## Требования

- Python 3.10 или новее (рекомендуется; используйте версию, поддерживаемую установленной сборкой PyTorch)
- [`факел`](https://pytorch.org/) (Достаточно сборки ЦП; установите сборку PyTorch, совместимую с вашим Python и платформой)
- [`сейфтензоры`](https://github.com/huggingface/safetensors)
- Графический интерфейс: `ткинтер` (обычно входит в состав установщиков Windows Python); `ttkbootstrap` является необязательным и предоставляет только тематический графический интерфейс

```баш
python -m pip install -r requirements.txt
```

Режим командной строки работает без `ткинтер` и `ttkbootstrap`. Включено `требования.txt` закрепляет полный набор зависимостей и включает в себя `ttkbootstrap`; если вам нужен только CLI, установите `факел` и `сейфтензоры` вместо этого совместим с вашей средой. Избегайте установки этих пакетов в Python Forge Neo, если вы намеренно не хотите изменить эту среду.

Слияние устанавливает количество внутриоперационных потоков ЦП PyTorch из `--потоки процессора` до загрузки тензорных полезных нагрузок. Каждый запуск выводит прошедшее время, рабочий набор начала/окончания процесса, пик, наблюдаемый сэмплером длительностью 50 мс, и пиковый рабочий набор времени жизни операционной системы, если он доступен. В Windows рабочий набор включает в себя как общие, так и частные резидентные страницы, включая страницы сопоставленных файлов; это не то же самое, что частный рабочий набор, и не является общей системной оперативной памятью. Пробоотборник может пропускать очень кратковременные пики, в то время как счетчик пиков срока службы Windows регистрирует высшую точку процесса.

## Использование

### ГРАФИЧЕСКИЙ ИНТЕРФЕЙС

Запустите скрипт без аргументов:

```баш
python Krea2_SVD_LoRA_Merger.py
```

1. Нажмите **Добавить файлы...** и выберите свои LoRA. Каждый файл проверяется на месте; проблемы объясняются в диалоговом окне.
2. Установить а **коэффициент** для каждого LoRA (по умолчанию `1,0`).
3. Выберите **выходной файл** (это должно закончиться в `.сейфтензоры`), **целевой ранг**, **dtype** и **Потоки ЦП** (по умолчанию 6; выберите 1–20).
4. Проверьте **текст активации** поле. Он заполняется автоматически из входных JSON-файлов LoRA, и вы можете его редактировать (см [Фордж Нео коляска JSON](#forge-neo-sidecar-json)).
5. Опционально отметьте галочкой **Сухой запуск** для проверки без сохранения или снятия отметки **Написать Forge Neo sidecar JSON** если вам это не нужно.
6. Нажмите **Запустить слияние** и следите за журналом. Нажмите **Отмена** чтобы остановить выполняющееся слияние (см [Отмена](#cancelling)).

Используйте селектор тем в правом нижнем углу, чтобы изменить внешний вид. Тема по умолчанию задается `ТЕМА_ПО УМОЛЧАНИЮ` в верхней части сценария.

### Командная строка

```bash
python Krea2_SVD_LoRA_Merger.py \
    --inputs style.safetensors character.safetensors detail.safetensors \
    --weights 1.0 0.7 0.5 \
    --output merged_r32.safetensors \
    --target-rank 32 \
    --dtype bf16
```

On Windows `cmd`, put it on one line or use `^` instead of `\`.

| Option | Default | Description |
| --- | --- | --- |
| `--inputs` | three example paths in the script | Input `.safetensors` LoRA files |
| `--output` | example path in the script | Output file (must end in `.safetensors`) |
| `--weights` | `1.0` for every input | One coefficient per input |
| `--target-rank` | `32` | Rank of the merged LoRA |
| `--cpu-threads` | `6` | PyTorch CPU intra-op threads (1–20); more threads may be slower on some CPUs |
| `--dtype` | `bf16` | Output dtype: `bf16`, `fp16` or `fp32` |
| `--dry-run` | off | Validate everything, write nothing |
| `--overwrite` | off | Allow replacing an existing output file and its sidecar JSON (the old JSON is kept as `.json.bak`) |
| `--no-pause` | off | Do not wait for Enter after an error (for scripts and `.bat` files) |
| `--activation-text` | collected from the input JSON files | Activation text for the output Forge Neo JSON (an empty string `""` writes an empty field) |
| `--no-forge-json` | off | Do not write the Forge Neo sidecar JSON |

The default paths near the top of the script (`D`, `DEFAULT_INPUTS`, `DEFAULT_OUTPUT`) are examples for a Windows setup; edit them to match your folders or always pass `--inputs` and `--output`.

## Forge Neo sidecar JSON

Forge Neo reads per-LoRA user metadata from a JSON file with the same name as the LoRA (`my_lora.safetensors` is paired with `my_lora.json`). By default the merger writes this file next to the output, built from the JSON files that sit next to the input LoRAs. Inputs without a JSON file are simply skipped.

| Field | How it is built |
| --- | --- |
| `description` | `Merged LoRA: ` followed by the input file names |
| `sd version` | The most common value among the input JSON files (`Unknown` is ignored). If none has one, the `FORGE_SD_VERSION_FALLBACK` constant at the top of the script is used (`krea`). |
| `activation text` | Phrases from all inputs, comma-separated, duplicates removed (case-insensitive), or your own text |
| `preferred weight` | The average of the positive values in the inputs. Inputs without a value are ignored, and `0` (Forge's "not set") is written if there are none. |
| `negative text` | Merged the same way as the activation text |
| `notes` | `Merged from: ` followed by the input names, plus the notes of the inputs |

A few things to know:

- The merge coefficients are already baked into the weights, so the averaged `preferred weight` is only a starting point. Test the result and adjust it.
- **Activation text in the GUI:** the field is refilled whenever you add or remove files. Text you typed yourself is not overwritten; the **Reload from JSON** button rebuilds it. If you clear the field on purpose, an empty activation text is saved. From the command line use `--activation-text`.
- **Existing JSON:** without `--overwrite` (or the overwrite checkbox) an existing JSON stops the run before any work is done. With overwrite, the old file is first copied to `<name>.json.bak`, because it may contain hand-written notes.
- If the LoRA was saved but the JSON could not be written, you get a warning rather than a failed run.
- Use `--no-forge-json` or untick the checkbox in the GUI to skip the file. A dry run prints the JSON that would be written.

## Cancelling

The **Cancel** button in the GUI is active while a merge is running. The merge stops at the next safe point: before each input file is loaded, between layers, and right before saving. Once the output starts being written, saving and validation are not interrupted, which keeps the atomic-save guarantee. A cancelled run writes neither the LoRA nor the JSON. Closing the window during a run asks for confirmation and cancels in the same way.

In command-line mode, press `Ctrl+C`; temporary files are cleaned up.
## How it works

For every layer, each LoRA contributes a weight delta

```
delta_i = w_i * (alpha_i / rank_i) * B_i @ A_i
```

Summing the deltas is equivalent to stacking the factors:

```
A_cat = [A_1; A_2; ...]   (concatenated along the rank dimension)
B_cat = [w_1*s_1*B_1, w_2*s_2*B_2, ...]   (concatenated along the rank dimension, s_i = alpha_i / rank_i)
sum(delta_i) = B_cat @ A_cat
```

The stacked rank is the sum of all input ranks, so it is re-packed to the target rank:

1. `B_cat = Qb Rb` and `A_cat^T = Qa Ra` (reduced QR).
2. `B_cat A_cat = Qb (Rb Ra^T) Qa^T`, and only the small core `Rb Ra^T` goes through SVD.
3. The top `k` singular values are kept and `sqrt(S)` is split between the new `A` and `B`.
4. The output `alpha` equals the target rank, so the effective scale is exactly 1.

If the stacked rank is smaller than the target rank, the extra rank is zero-padded.

The reported per-layer metric is the relative Frobenius norm of the discarded singular-value tail: `||S[k:]||₂ / ||S||₂` (with zero returned when `||S||₂ = 0`). The summary reports the mean and maximum of this metric across layers. It estimates only rank-truncation loss; it does not include final output-dtype quantization error.

## Supported LoRA formats

- Key styles: `lora_down` / `lora_up` (kohya-style) and `lora_A` / `lora_B` (PEFT/diffusers-style, including `.default`).
- **LoKr input conversion:** direct full-matrix `lokr_w1` + `lokr_w2` pairs are converted internally to Kohya-style LoRA factors; the output then follows the same SVD merge path as ordinary LoRAs.
- Alpha keys: `.alpha` or `.lora_alpha`. Missing alpha is treated as `alpha = rank` (scale 1).
- Linear (2D) layers only.
- Layer names must match between LoRAs to be summed. A layer that exists in only one LoRA is kept and re-packed on its own.
- The output reuses the key style of the first LoRA that contains the layer.

## Limitations and notes

- **LoKr:** direct full-matrix `lokr_w1` / `lokr_w2` modules are detected automatically and converted to ordinary SVD LoRA at the selected target rank before merging. Conversion uses the factor SVDs and does not materialize the full Kronecker matrix. For direct full-matrix LyCORIS LoKr, alpha is not used as an extra scale; the converted LoRA is written with alpha equal to the target rank. Decomposed LoKr (`lokr_w*_a/b`), Tucker (`lokr_t2`) and LoHa are not supported and are rejected explicitly. Conv (4D) LoRA remains unsupported.
- **Alpha without tensors:** if a LoRA has no alpha tensors but was trained with `alpha != rank` (the value is then only in `ss_network_alpha` metadata or in a training config), the script assumes scale 1 and prints a warning. Check such files by testing the result.
- **Memory (standard mode):** all input LoRAs are loaded into RAM at once, in their stored dtype. The separate matrix-free mode reads only the current module's source tensors, but the output factors still accumulate in RAM.
- **Metadata:** the output is labeled `modelspec.architecture = krea2/lora` and `ss_base_model_version = krea2`. Along with it the file records the merge inputs, coefficients, target rank and dtype (`merge_*` keys). Change these in `merge_loras()` if you use the script for another model.
- Always test a merged LoRA at low strength first; merged deltas can behave differently from stacked LoRAs, especially at high coefficients.

## Troubleshooting

| Problem | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: tkinter` | Your Python has no Tk. Use the command-line mode, or install Python with Tcl/Tk support. |
| GUI looks plain | `ttkbootstrap` is not installed: `pip install ttkbootstrap`. |
| `Unrecognized tensors ...` | The file contains unsupported or mixed tensor keys (for example LoHa, decomposed/Tucker LoKr, or extra data). Supported full-matrix LoKr is converted automatically. The script stops rather than dropping unknown data. |
| `Dimension mismatch ...` | The LoRAs were trained for different base models or architectures. |
| `fp16 overflow at ...` | Use `bf16` or `fp32` as the output dtype. |
| `Output exists ...` | Pick another file name or pass `--overwrite` (or tick the checkbox in the GUI). |



## Matrix-free joint LoKr + LoRA SVD (approximate)

The separate **Matrix-free joint SVD** mode combines direct-matrix LoKr terms and
ordinary LoRA terms through an implicit linear operator. It does not construct
the Kronecker matrix or expand LoKr into full-rank A/B factors. Modules are
processed one at a time, so input payloads are not all retained in RAM together.

This mode uses randomized SVD and is therefore approximate. Its reported
relative residual is estimated with random probe vectors; it is not the exact
Frobenius truncation error reported by the standard QR/SVD path. Negative merge
coefficients are supported for both LoKr and ordinary LoRA contributions.

CLI example:

~~~bash
python Krea2_SVD_LoRA_Merger.py --inputs lokr.safetensors style.safetensors ^
  --weights 1.0 -0.35 --output merged_matrix_free.safetensors ^
  --target-rank 32 --lokr-matrix-free --lokr-svd-oversample 8 ^
  --lokr-svd-power-iterations 2 --lokr-workspace-max-gb 2.0
~~~

В графическом интерфейсе есть отдельный флажок для этого режима. Его нельзя комбинировать с
**Сохраняйте полный ранг LoKr до финального SVD**. Предварительная проверка рабочего пространства — это
только консервативная оценка алгоритма и рабочего пространства: она исключает входной тензор
полезные нагрузки, выходные тензоры и память времени выполнения Python/PyTorch. Сохраняйте лимит
консервативен в системах с 16 ГБ оперативной памяти. Начните с избыточной выборки по умолчанию и
настройки итерации мощности; их увеличение увеличивает время выполнения и использование памяти.

Режим без матриц в настоящее время поддерживает тот же прямой полноматричный LoKr
макет как существующий преобразователь (lokr_w1 + lokr_w2) и обычный 2D LoRA
Пары А/В. Остаются разложенные/Таккер LoKr, LoHa и сверточные факторы
неподдерживаемый.

## Точное сохранение LoKr (единственный конечный SVD)

Необязательно **Сохраняйте полный ранг LoKr до финального SVD** режим избегает
усѵчение промежуточного ранга, используемое преобраазователм LoKr пѾ умѾчанию. Прямой
Полноматричные фактѾры LoKr Ѱзлагаются в тѾчные обычные фактѾры A/B, заЂем
объединяется с другими выбранными LoRA и сжимается один раз обычным способом
новый QR + новый SVD-каскад.

Этот режим может использовать значительно больше оперативной памяти. Перед загрузкой тензорных полезных нагрузок
славные офсенки хронилище факторов Float32 A/B на сайте SafeTensors.
Графический интерфейс использует консервативный предел оценки в 2 ГиБ. Пример CLI:

```баш
python Krea2_SVD_LoRA_Merger.py --выходные данные lokr_a.safetensors lokr_b.safetensors ordinal.safetensors --выходные данные merged_fullrank.safetensors --target-rank 32 --lokr-full-rank --lokr-max-factor-gb 2.0
```

Оценка охватывает расшиЀенные кѾэффициѵты LoKr A/B, а не каждый веменный тензор или
программа рабы Python/PyTorch. Сохраняйте консервативный лимит на системах с 16 ГБ
ОЗУ; не увеличивайте его, если не доступно достаточно физической памяти. The
режимом по умолчанию остается преобразование Ѐнга LoKr с блнизкй памятью.


## Режимные слоения LoKr

| Режим | Процесс | Усечения | Сообщение об ошибке |
|---|---|---|---|
| По умолчанию | LoKr → ранг `целевой ранг` → слияние → `целевой ранг` | **Два** | Ошибка 2-го этапа + верхняя граница общей ошибки |
| `--локр-полный-ранг` | LoKr (без образования) → светлое → `целевой ранг` | Один | Точная ошибка (на основе отброшенных сингулярных значений) |
| `--lokr-matrix-free` | Рандомизированный СВД суммы (без расширения произведений Кронекера) | Один (приблизительно) | Остаточная оценка методом случайной выборки |

**ГРАФИЧЕСКИЙ ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ.** При добавлении файла LoKr прааграмма оодин раз за ссанс сспрашиваѵт, следуѵт ли включитьь оѿцю «Безматричное сединение SVD» (кторая выполяѵт д� Эт� `--локр-полный-ранг` режим не предлагается в диалоговом окне (хотя его можно включить вручную); `AUTO_LOKR_OPTION` постоянно контролирует, какой режим предлагается.

### Понимание значений ошибок

* **«Относительная ошибка Фробениуса только из-за усечения ранга»** (плотный режим) рассчитывается относительно *уже усеченный* сумма. В режиме LoKr по работе это знячене **не включает** потери, понесенные на первом этапе, что привело к недооценке общей погрешности (например, 0,244 в синтетическом тесте по сравнению с фактичес�
* **«Комбинированная ошибка… ВЕРХНЯЯ граница»** представляет собой верхнюю границу относительной погрешности —определяется как `‖точная сумма − результат‖ / ‖точная сумма‖`—получено с использованием неравенства треугольника. Это намеренно грубая оценка (на основе 200 синтетических тестов отношение границы к фактическому значению составляет: мин. 1,00, медиана 1,81, ма
Этот режим использует рандомизированный SVD и поэтому является приблизительным. Сообщается, что **"Относительная остаточная оценка случайного зонда"** (без матриц) аппроксимирует относительную ошибку Фробениуса *результирующие факторы по отношению к истинному оператору* (с учетом как рандомизации, так и усечения). Это зашумленная оценка (с дисперсией ≈10% при использовании 8 зондов); она не является ни формальной границей, ни оптимумом Эккарта–Янга.
* Порог 0,01, упомянутый в предупреждении, является всего лишь диагностической подсказкой, а не формальным критерием качества.

## Альфа-семантика LoKr — СТАТУС: НЕ ПОЛНОСТЬ ПРОФЕРЕНО

**Как функционирует код.** Полномочный LoKr (`lokr_w1` + `lokr_w2`) применяется как `крон(w1, w2)` по шкале 1; `альфа` тензор считывается, но **не применяется**. Разложившиеся варианты (`lokr_w*_a/_b`, `lokr_t2`) не поддерживаются и вызовут ошибку (как `альфа`/`классифицировать` значимы для этих типов).

**Основа для данной информации.**

| Нет. | Источник | Содержание | Надежность |
|---|---|---|---|
| 1 | Копии `набор информации/modeli/lokr.py` (из ai-toolkit, Hugging Face Spaces), программное обеспечение «в значительный список фанчий на коммите LyCORIS eb46009…» | `эли self.use_w2 и self.use_w1: альфа = lora_dim # изользование масштаба = 1`; за которым следует `масштаб = альфа / lora_dim`; `make_kron` затем умножает на `шкала` | Производная копия, а не первоисточник | | 2 | Выпуск shootthesound/Fizgig #304 | Ссылки `lycoris/модули/lokr.py`, стоки 209–218: содержит тот же комментантай, "используйте машштаб = 1"; также утвержддат, что ComfyUI ingnoriruetet `альфа` для полноматричного случая. | Вторичный счет |
| 3 | ПЭФТ `LoKrLayer.update_layer` | Наборы `самомасштабирование[адаптер] = альфа / р` для всех слоев LoKr; это **неясно** применимо ли это к случаю полной матрицы. | Неполный |

Что НЕ было проверено:коэффициент *= альфа/ранг

* «Предварительная проверка рабочего пространства» — это *алгоритмический* эвристика памяти; она не гарантирует, что процесс поместится в оперативную память. Он не учитывает входные тензоры, выходные тензоры, среду выполнения Python/PyTorch или временные копии.
* В «плотном» режиме входные файлы считываются полностью (`загрузить_файл`); в режиме «без матриц» они считываются по одному модулю за раз. Потоковое чтение не реализовано для плотного режима.
Полноматричные фактѾры LoKr Ѱзлагаются в тѾчные обычные фактѾры A/B, заЂем `_kronecker_svd` (после исправления): пиковое использование памяти составляет приблизительно 1,1–1,5× размера выходных матриц A+B (при полном ранге), что ниже ~2,5× 

## Известные ограничения

* Объединение контрольных тѾчѵK Кохья и PEFT рабѾтаѵт тоько в том случае, если имена модулей (Ѿсновы) совпадают. Автоматической нормализации имен не существует (например, сопоставление между `lora_unet_…` и `диффузионная_модель.…`).
* Если входной тензор содержит элементы порядка ≤1e-19 (в `float32`), они могут рассматриваться как ноль во время вычислений ядра; в таких случаях ошибка, возникающая из-за нулевой нормы, сообщается как `0,0`.
