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
    
    fetch(`/start_tracking/${category}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Redirecting to simulator page...');
            window.location.href = `/simulator/${category}`;
        })
        .catch(error => {
            console.error('Error:', error);
            button.disabled = false;
            button.textContent = category.charAt(0).toUpperCase() + category.slice(1);
            alert(`Failed to start tracking for ${category}. Please try again.`);
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