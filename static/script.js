document.addEventListener('DOMContentLoaded', function () {
    const anchorLinks = document.querySelectorAll('nav a[href^="#"]');

    anchorLinks.forEach(link => {
        link.addEventListener('click', function (event) {
            event.preventDefault();
            const targetId = this.getAttribute('href').slice(1);
            const target = document.getElementById(targetId);
            if (target) {
                window.scrollTo({ top: target.offsetTop - 80, behavior: 'smooth' });
            }
        });
    });

    // Camera control logic
    const startBtn = document.getElementById('start-cam-btn');
    const stopBtn = document.getElementById('stop-cam-btn');
    const videoContainer = document.getElementById('video-container');
    const videoFeed = document.getElementById('video-feed');

    const cameraStatus = document.getElementById('camera-status');

    if (startBtn && stopBtn && videoContainer && videoFeed) {
        videoFeed.addEventListener('load', function () {
            if (cameraStatus) {
                cameraStatus.textContent = 'Live';
            }
        });

        videoFeed.addEventListener('error', function () {
            if (cameraStatus) {
                cameraStatus.textContent = 'Stream error';
            }
        });

        startBtn.addEventListener('click', function (event) {
            event.preventDefault();
            videoContainer.style.display = 'grid';
            videoFeed.style.display = 'block';
            videoFeed.alt = 'Starting camera stream...';
            if (cameraStatus) {
                cameraStatus.textContent = 'Starting...';
            }
            videoFeed.removeAttribute('src');
            videoFeed.src = '/frame?cache=' + Date.now();

            if (videoFeed.complete && videoFeed.naturalWidth !== 0) {
                if (cameraStatus) {
                    cameraStatus.textContent = 'Live';
                }
            }

            videoContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });

        videoFeed.addEventListener('load', function () {
            if (cameraStatus) {
                cameraStatus.textContent = 'Live';
            }
            setTimeout(function () {
                videoFeed.src = '/frame?cache=' + Date.now();
            }, 1000);
        });

        videoFeed.addEventListener('error', function () {
            if (cameraStatus) {
                cameraStatus.textContent = 'Stream error';
            }
        });

        stopBtn.addEventListener('click', function () {
            videoFeed.removeAttribute('src');
            videoFeed.style.display = 'none';
            videoContainer.style.display = 'none';
            if (cameraStatus) {
                cameraStatus.textContent = 'Stopped';
            }
        });
    }
});
