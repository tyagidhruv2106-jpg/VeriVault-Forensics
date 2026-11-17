document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const prompt = dropZone ? dropZone.querySelector('.drop-zone__prompt') : null;
    const form = document.getElementById('uploadForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                prompt.textContent = fileInput.files[0].name;
            } else {
                prompt.textContent = "Drag & Drop or Click to Upload Document";
            }
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drop-zone--over');
        });

        ['dragleave', 'dragend'].forEach(type => {
            dropZone.addEventListener(type, (e) => {
                dropZone.classList.remove('drop-zone--over');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                prompt.textContent = e.dataTransfer.files[0].name;
            }
            dropZone.classList.remove('drop-zone--over');
        });
    }

    if (form) {
        form.addEventListener('submit', () => {
            if(fileInput.files.length > 0) {
                analyzeBtn.classList.add('hidden');
                loadingIndicator.classList.remove('hidden');
            }
        });
    }
});
