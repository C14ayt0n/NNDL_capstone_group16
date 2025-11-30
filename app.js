// глобальная ссылка на модель
let model = null;

// !!! ВАЖНО: сюда нужно подставить реальные mean/std,
// которые ты получил в ноутбуке encode0 для
// [total_calories, total_mass, total_fat, total_carb, total_protein]
const NUTRITION_MEAN = [
  /* total_calories_mean */,
  /* total_mass_mean */,
  /* total_fat_mean */,
  /* total_carb_mean */,
  /* total_protein_mean */
];

const NUTRITION_STD = [
  /* total_calories_std */,
  /* total_mass_std */,
  /* total_fat_std */,
  /* total_carb_std */,
  /* total_protein_std */
];

// загрузка модели при старте страницы
window.addEventListener('load', async () => {
  const statusEl = document.getElementById('status');

  try {
    statusEl.textContent = 'Загружается модель...';

    // model.json и group1-shard*.bin лежат в той же папке
    model = await tf.loadGraphModel('./model.json');

    statusEl.textContent = 'Модель загружена. Выберите изображение.';
  } catch (err) {
    console.error(err);
    statusEl.textContent =
      'Ошибка загрузки модели. Проверьте путь к model.json.';
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
    statusEl.textContent = 'Модель ещё не загружена.';
    return;
  }

  const reader = new FileReader();

  // когда файл прочитан — показываем превью и запускаем инференс
  reader.onload = () => {
    const img = document.getElementById('preview');
    if (!img) return;

    img.onload = async () => {
      statusEl.textContent = 'Выполняется предсказание...';

      try {
        const preds = await runInferenceOnImage(img);
        updateResults(preds);
        statusEl.textContent = 'Готово.';
      } catch (err) {
        console.error(err);
        statusEl.textContent = 'Ошибка при предсказании. См. консоль.';
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
    let x = tf.browser.fromPixels(img).toFloat(); // [H, W, 3]

    // 2) resize до 224x224 (как в модели)
    x = tf.image.resizeBilinear(x, [224, 224], true);

    // 3) НЕ делим на 255, т.к. внутри EfficientNetV2 уже есть Rescaling(1/255)
    // при include_preprocessing=True в Keras-модели

    // 4) добавление batch dimension
    x = x.expandDims(0); // [1, 224, 224, 3]

    // 5) вызов выходного тензора модели (output_0 / Identity:0)
    const y = model.execute(x, 'Identity:0'); // [1, 5]

    // y содержит нормированные значения (z-скоры)
    const vals = y.dataSync(); // Float32Array длиной 5

    // денормализация по формуле y = z * std + mean
    const denorm = new Array(vals.length);
    for (let i = 0; i < vals.length; i++) {
      denorm[i] = vals[i] * NUTRITION_STD[i] + NUTRITION_MEAN[i];
    }

    // порядок признаков: [calories, mass, fat, carb, protein]
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
