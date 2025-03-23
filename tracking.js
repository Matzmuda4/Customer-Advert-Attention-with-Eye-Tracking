let currentCategory = null;
let currentImage = null;
let statusText = null;
let currentElement = null;
let lastHoverTime = null;
let elementTimes = {};

function startTracking(category) {
    currentCategory = category;
    const button = event.target;
    button.disabled = true;
    button.textContent = 'Starting...';
    
    fetch(`http://localhost:8080/start_tracking/${category}`)
        .then(response => response.json())
        .then(data => {
            window.location.href = `http://localhost:8080/simulator/${category}`;
        })
        .catch(error => {
            console.error('Error:', error);
            button.disabled = false;
            button.textContent = 'Error - Try Again';
        });
}

function initializeTracking() {
    currentImage = document.getElementById('currentImage');
    statusText = document.getElementById('statusText');
    
    currentImage.addEventListener('mousemove', function(event) {
        const rect = currentImage.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        updateTracking(x, y);
    });
}

function updateTracking(x, y) {
    const currentTime = Date.now() / 1000;  // Convert to seconds
    
    fetch(`http://localhost:8080/track/${currentCategory}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            x: x,
            y: y,
            timestamp: currentTime
        })
    })
    .then(response => response.json())
    .then(data => {
        statusText.textContent = `Current element: ${data.element}\nPosition: (${Math.round(x)}, ${Math.round(y)})\nTime on element: ${data.time.toFixed(2)}s`;
    });
}

window.onload = initializeTracking; 