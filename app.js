// глобальная ссылка на модель
let model = null;

// статистики таргетов из ноутбука encode0
// порядок: [total_calories, total_mass, total_fat, total_carb, total_protein]
const NUTRITION_MEAN = [
  257.7,   // calories mean
  214.42,  // mass mean
  12.97,   // fat mean
  19.28,   // carb mean
  18.26    // protein mean
];

const NUTRITION_STD = [
  211.42,  // calories std
  153.17,  // mass std
  13.72,   // fat std
  16.17,   // carb std
  20.14    // protein std
];

// функция, которая строит корректный url до model.json
// работает и локально, и на github pages (с базовым путём вида /NNDL_capstone_group16/)
function getModelUrl() {
  const baseUrl = window.location.href.replace(/index\.html?$/, '');
  return new URL('model.json', baseUrl).toString();
}

// загрузка модели при старте страницы
window.addEventListener('load', async () => {
  const statusEl = document.getElementById('status');

  try {
    statusEl.textContent = 'загружается модель...';

    const modelUrl = getModelUrl();
    console.log('загружаем модель из:', modelUrl);

    // model.json и group1-shard*.bin лежат в той же папке, что и index.html
    model = await tf.loadGraphModel(modelUrl);

    statusEl.textContent = 'модель загружена. выберите изображение.';
  } catch (err) {
    console.error('ошибка загрузки модели:', err);
    statusEl.textContent =
      'ошибка загрузки модели. проверьте путь к model.json.';
  }

  const fileInput = document.getElementById('file-input');
  if (fileInput) {
    fileInput.addEventListener('change', handleFileChange);
  }
});

// обработка выбора файла
function handleFileChange(event) {
  const file = event.target.files[0];
  if (!file) return;

  const statusEl = document.getElementById('status');

  if (!model) {
    statusEl.textContent = 'модель ещё не загружена.';
    return;
  }

  const reader = new FileReader();

  // когда файл прочитан — показываем превью и запускаем инференс
  reader.onload = () => {
    const img = document.getElementById('preview');
    if (!img) return;

    img.onload = async () => {
      statusEl.textContent = 'выполняется предсказание...';

      try {
        const preds = await runInferenceOnImage(img);
        updateResults(preds);
        statusEl.textContent = 'готово.';
      } catch (err) {
        console.error('ошибка при предсказании:', err);
        statusEl.textContent = 'ошибка при предсказании. см. консоль.';
      }
    };

    img.src = reader.result;
  };

  reader.readAsDataURL(file);
}

// инференс: препроцессинг изображения и вызов модели
function runInferenceOnImage(img) {
  return tf.tidy(() => {
    // 1) из пикселей -> float32
    let x = tf.browser.fromPixels(img).toFloat(); // [H, w, 3]

    // 2) resize до 224x224 (как в исходной keras-модели) [web:7]
    x = tf.image.resizeBilinear(x, [224, 224], true);

    // 3) не делим на 255, efficientnetv2 внутри сама делает rescaling(1/255)

    // 4) добавляем batch dimension
    x = x.expandDims(0); // [1, 224, 224, 3]

    // 5) вызываем выходной тензор модели (output_0 / identity:0)
    const y = model.execute(x, 'Identity:0'); // [1, 5]

    // y содержит нормированные значения (z-скоры)
    const vals = y.dataSync(); // Float32Array длиной 5
    console.log('raw z-scores:', vals);

    // денормализация: y = z * std + mean
    const denorm = new Array(vals.length);
    for (let i = 0; i < vals.length; i++) {
      denorm[i] = vals[i] * NUTRITION_STD[i] + NUTRITION_MEAN[i];
    }
    console.log('denormalized:', denorm);

    // порядок признаков: [calories, mass, fat, carb, protein] [web:7]
    const calories = round1(denorm[0]);
    const mass = round1(denorm[1]);
    const fat = round1(denorm[2]);
    const carbs = round1(denorm[3]);
    const protein = round1(denorm[4]);

    return { calories, mass, fat, carbs, protein };
  });
}

// округление до 1 знака после запятой
function round1(x) {
  return Math.round(x * 10) / 10;
}

// вывод результатов в интерфейс
function updateResults(p) {
  const caloriesEl = document.getElementById('calories');
  const fatEl = document.getElementById('fat');
  const carbsEl = document.getElementById('carbs');
  const proteinEl = document.getElementById('protein');
  const massEl = document.getElementById('mass'); // опционально, если есть такой элемент

  if (caloriesEl) caloriesEl.textContent = p.calories;
  if (fatEl) fatEl.textContent = p.fat;
  if (carbsEl) carbsEl.textContent = p.carbs;
  if (proteinEl) proteinEl.textContent = p.protein;
  if (massEl && typeof p.mass !== 'undefined') massEl.textContent = p.mass;
}
