let model = null;

// загрузка модели
window.addEventListener('load', async () => {
  const statusEl = document.getElementById('status');
  try {
    statusEl.textContent = 'Загружается модель...';
    // model.json и group1-shard*.bin лежат в той же папке
    model = await tf.loadGraphModel('./model.json');
    statusEl.textContent = 'Модель загружена. Выберите изображение.';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Ошибка загрузки модели. Проверьте путь к model.json.';
  }

  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', handleFileChange);
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
  reader.onload = () => {
    const img = document.getElementById('preview');
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

// препроцессинг как в ноутбуке encode0: Rescaling(1/255) перед EfficientNetV2 с include_preprocessing=True
function runInferenceOnImage(img) {
  return tf.tidy(() => {
    // 1) из пикселей -> float32
    let x = tf.browser.fromPixels(img).toFloat(); // [H,W,3]

    // 2) resize до 224x224 (как в модели)
    x = tf.image.resizeBilinear(x, [224, 224], true);

    // 3) масштабирование в [0,1] (Rescaling(1/255))
    x = x.div(255.0);

    // 4) добавление batch dimension
    x = x.expandDims(0); // [1,224,224,3]

    // 5) явный вызов выходного тензора модели
    // в model.json выход называется Identity:0 (output_0)
    const y = model.execute(x, 'Identity:0'); // [1,5]

    const vals = y.dataSync(); // Float32Array длиной 5

    // порядок как в dish_ingredients / ноутбуке:
    // [total_calories, total_fat, total_carb, total_protein, ...]
    const calories = round1(vals[0]);
    const fat      = round1(vals[1]);
    const carbs    = round1(vals[2]);
    const protein  = round1(vals[3]);

    return { calories, fat, carbs, protein };
  });
}

function round1(x) {
  return Math.round(x * 10) / 10;
}

function updateResults(p) {
  document.getElementById('calories').textContent = p.calories;
  document.getElementById('fat').textContent      = p.fat;
  document.getElementById('carbs').textContent    = p.carbs;
  document.getElementById('protein').textContent  = p.protein;
}
