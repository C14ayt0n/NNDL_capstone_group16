let model = null;

// Загрузка модели при старте страницы
window.addEventListener('load', async () => {
  const statusEl = document.getElementById('status');
  try {
    statusEl.textContent = 'Загружается модель...';
    // model.json и shard*.bin лежат в той же папке
    model = await tf.loadGraphModel('./model.json');
    statusEl.textContent = 'Модель загружена. Выберите изображение.';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Ошибка загрузки модели. Проверьте путь к model.json.';
  }

  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', handleFileChange);
});

// Обработка выбора файла
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
        const predictions = await runInferenceOnImageElement(img);
        updateResults(predictions);
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

// Препроцессинг и инференс для <img>
async function runInferenceOnImageElement(img) {
  // Внимание: если датасет/обучение использовали другой препроцессинг,
  // здесь нужно воспроизвести тот же самый пайплайн.
  // Для EfficientNetV2 обычно: масштабирование к [0,1], затем нормализация.
  return tf.tidy(() => {
    // 1. fromPixels -> [H, W, 3]
    let tensor = tf.browser.fromPixels(img).toFloat();

    // 2. resize до 224x224
    tensor = tf.image.resizeBilinear(tensor, [224, 224], true);

    // 3. Масштабирование в [0,1]
    tensor = tensor.div(255.0);

    // 4. NCHW -> добавляем batch dimension: [1, 224, 224, 3]
    tensor = tensor.expandDims(0);

    // 5. Вызов модели
    const output = model.execute
      ? model.execute(tensor)
      : model.predict(tensor);

    // Ожидаем выход формы [1, 5]
    const values = output.dataSync(); // Float32Array длиной 5

    // Берём первые 4: [calories, protein, fat, carbs]
    const calories = round1(values[0]);
    const protein  = round1(values[1]);
    const fat      = round1(values[2]);
    const carbs    = round1(values[3]);

    return { calories, protein, fat, carbs };
  });
}

// Округление до 1 знака
function round1(x) {
  return Math.round(x * 10) / 10;
}

// Обновление DOM с результатами
function updateResults(pred) {
  document.getElementById('calories').textContent = pred.calories;
  document.getElementById('protein').textContent  = pred.protein;
  document.getElementById('fat').textContent      = pred.fat;
  document.getElementById('carbs').textContent    = pred.carbs;
}
