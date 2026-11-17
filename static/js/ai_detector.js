// VeriVault - Universal AI Image & Manipulation Detector Frontend Controller

document.addEventListener('DOMContentLoaded', () => {
    // Check if URL hash specifies #metrics
    if (window.location.hash === '#metrics') {
        switchTab('metrics');
    }

    const dropZone = document.getElementById('aiDropZone');
    const fileInput = document.getElementById('aiFileInput');
    const promptText = document.getElementById('aiDropPrompt');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                promptText.textContent = `Selected: ${fileInput.files[0].name}`;
                dropZone.classList.add('drop-zone--over');
                hideError();
            } else {
                promptText.textContent = "Drag & Drop or Click to Upload Image";
                dropZone.classList.remove('drop-zone--over');
            }
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drop-zone--over');
        });

        ['dragleave', 'dragend'].forEach(type => {
            dropZone.addEventListener(type, () => {
                dropZone.classList.remove('drop-zone--over');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                promptText.textContent = `Selected: ${e.dataTransfer.files[0].name}`;
                hideError();
            }
            dropZone.classList.remove('drop-zone--over');
        });
    }
});

// Tab Switcher
function switchTab(tabName) {
    const tabDetector = document.getElementById('tabDetector');
    const tabMetrics = document.getElementById('tabMetrics');
    const btnDetector = document.getElementById('tabBtnDetector');
    const btnMetrics = document.getElementById('tabBtnMetrics');

    if (tabName === 'metrics') {
        tabDetector.classList.remove('active');
        tabMetrics.classList.add('active');
        btnDetector.classList.remove('active');
        btnMetrics.classList.add('active');
        window.location.hash = 'metrics';
    } else {
        tabMetrics.classList.remove('active');
        tabDetector.classList.add('active');
        btnMetrics.classList.remove('active');
        btnDetector.classList.add('active');
        history.replaceState(null, null, ' ');
    }
}

function showError(msg) {
    const errorBox = document.getElementById('aiErrorBox');
    errorBox.textContent = msg;
    errorBox.classList.remove('hidden');
}

function hideError() {
    const errorBox = document.getElementById('aiErrorBox');
    errorBox.classList.add('hidden');
}

// Quick Sample Testing
function testSample(sampleName) {
    const loading = document.getElementById('aiLoadingIndicator');
    const btn = document.getElementById('runAiDetectBtn');
    const results = document.getElementById('aiResultsSection');

    hideError();
    loading.classList.remove('hidden');
    btn.classList.add('hidden');
    results.classList.add('hidden');

    fetch('/api/detect-ai-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_name: sampleName })
    })
    .then(res => res.json())
    .then(data => {
        loading.classList.add('hidden');
        btn.classList.remove('hidden');
        if (data.error) {
            showError(data.error);
        } else {
            renderResults(data);
        }
    })
    .catch(err => {
        loading.classList.add('hidden');
        btn.classList.remove('hidden');
        showError(`Request failed: ${err.message}`);
    });
}

// User File Submission
function submitAiImage() {
    const fileInput = document.getElementById('aiFileInput');
    if (!fileInput.files || fileInput.files.length === 0) {
        showError("Please select or drop an image file first, or click one of the benchmark samples.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);

    const loading = document.getElementById('aiLoadingIndicator');
    const btn = document.getElementById('runAiDetectBtn');
    const results = document.getElementById('aiResultsSection');

    hideError();
    loading.classList.remove('hidden');
    btn.classList.add('hidden');
    results.classList.add('hidden');

    fetch('/api/detect-ai-image', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loading.classList.add('hidden');
        btn.classList.remove('hidden');
        if (data.error) {
            showError(data.error);
        } else {
            renderResults(data);
        }
    })
    .catch(err => {
        loading.classList.add('hidden');
        btn.classList.remove('hidden');
        showError(`Inference request failed: ${err.message}`);
    });
}

// Render Results on UI
function renderResults(data) {
    const results = document.getElementById('aiResultsSection');
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictIcon = document.getElementById('verdictIcon');
    const verdictText = document.getElementById('verdictText');
    const confBadge = document.getElementById('confidenceBadge');
    
    const probRealLabel = document.getElementById('probRealLabel');
    const probAiLabel = document.getElementById('probAiLabel');
    const probBarReal = document.getElementById('probBarReal');
    const probBarAi = document.getElementById('probBarAi');

    const viewOrig = document.getElementById('viewOriginalImg');
    const viewGradcam = document.getElementById('viewGradcamImg');
    const viewEla = document.getElementById('viewElaImg');
    const dimText = document.getElementById('imgDimensions');
    const diagnosisList = document.getElementById('diagnosisList');

    // Verdict Styling
    const isAi = data.verdict_badge === 'AI_MANIPULATED';
    verdictBanner.className = `result-banner ${isAi ? 'ai' : 'real'}`;
    verdictIcon.textContent = isAi ? '⚠️' : '🛡️';
    verdictText.textContent = data.verdict;
    confBadge.textContent = `${data.confidence_score}% Confidence`;

    // Probability Bars
    probRealLabel.textContent = `Authentic Probability: ${data.prob_real}%`;
    probAiLabel.textContent = `AI / Tampered Probability: ${data.prob_ai}%`;
    probBarReal.style.width = `${data.prob_real}%`;
    probBarAi.style.width = `${data.prob_ai}%`;

    // Visual Triad
    viewOrig.src = data.original_image;
    viewGradcam.src = data.gradcam_image;
    viewEla.src = data.ela_image;
    dimText.textContent = `Dimensions: ${data.dimensions || '224 x 224'}`;

    // Diagnosis Bullet Points
    diagnosisList.innerHTML = '';
    if (data.forensic_explanation && Array.isArray(data.forensic_explanation)) {
        data.forensic_explanation.forEach(point => {
            const li = document.createElement('li');
            li.innerHTML = `<span class="bullet">▸</span> <span>${point}</span>`;
            diagnosisList.appendChild(li);
        });
    }

    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

