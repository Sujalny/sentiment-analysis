const form = document.querySelector('#analyze-form');
const input = document.querySelector('#text');
const submit = document.querySelector('#submit');
const error = document.querySelector('#error');
function updateCount() { document.querySelector('#count').textContent = `${input.value.length.toLocaleString()} / 5,000`; }
input.addEventListener('input', updateCount);
document.querySelector('#clear').addEventListener('click', () => { input.value = ''; updateCount(); input.focus(); });
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => { input.value = button.dataset.example; updateCount(); input.focus(); }));
form.addEventListener('submit', async event => {
  event.preventDefault();
  error.hidden = true;
  submit.disabled = true;
  submit.textContent = 'Analyzing…';
  document.querySelector('#result').setAttribute('aria-busy', 'true');
  document.querySelector('#prediction').hidden = true;
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: input.value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed. Please try again.');
    document.querySelector('#empty').hidden = true;
    document.querySelector('#prediction').hidden = false;
    const title = document.querySelector('#sentiment');
    title.textContent = data.uncertain ? `${data.sentiment} · uncertain` : data.sentiment;
    title.className = data.sentiment;
    const scores = document.querySelector('#scores');
    scores.replaceChildren();
    for (const label of ['positive', 'neutral', 'negative']) {
      const row = document.createElement('div');
      row.className = 'score';
      const caption = document.createElement('div');
      caption.textContent = `${label[0].toUpperCase() + label.slice(1)} · ${(data.scores[label] * 100).toFixed(1)}%`;
      const track = document.createElement('div');
      track.className = 'track';
      const fill = document.createElement('div');
      fill.className = `fill ${label}`;
      fill.style.width = `${data.scores[label] * 100}%`;
      track.append(fill); row.append(caption, track); scores.append(row);
    }
  } catch (failure) {
    error.textContent = failure.message;
    error.hidden = false;
  } finally {
    submit.disabled = false;
    submit.textContent = 'Analyze sentiment ↗';
    document.querySelector('#result').setAttribute('aria-busy', 'false');
  }
});
