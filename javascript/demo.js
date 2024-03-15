let isActive = false;
let openLabels = [];
let fullscreenButton;
let sliderLoaded = false;
let vidLength = 0;
let vidFps = 0;


function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

function getRealElement(selector, get_input = false) {
    let elem = gradioApp().getElementById(selector);
    let output = elem;
    if (elem) {
        let child = elem.querySelector('#' + selector);
        if (child) {
            output = child;
        }
    }
    if (get_input) {
        return output.querySelector('input');
    }
    return output;
}

function filterArgs(argsCount, arguments) {
    let args_out = [];
    if (arguments.length >= argsCount && argsCount !== 0) {
        for (let i = 0; i < argsCount; i++) {
            args_out.push(arguments[i]);
        }
    }
    return args_out;
}

function update_slider() {
    console.log("Input args: ", arguments);
    configureSlider(arguments[4], arguments[3]);
    return filterArgs(6, arguments);
}

function configureSlider(videoLength, fps) {
    if (vidFps === fps && vidLength === videoLength) {
        console.log('configureSlider', 'videoLength and fps are the same');
        return;
    }
    if (videoLength === 0) {
        console.log('configureSlider', 'videoLength is 0');
        return;
    }
    console.log('configureSlider, re-creating slider.', 'videoLength', videoLength, 'fps', fps);
    vidFps = fps;
    vidLength = videoLength;
    let connectSlider2 = document.getElementById('frameSlider');
    let endTimeLabel = document.getElementById('endTimeLabel');
    let nowTimeLabel = document.getElementById('nowTimeLabel');
    if (sliderLoaded) {
        try {
            connectSlider2.noUiSlider.destroy();
        } catch (e) {
            console.log('configureSlider', 'noUiSlider.destroy failed', e);
        }
    }


    let midPoint = Math.floor(videoLength / 2);

    noUiSlider.create(connectSlider2, {
        start: [0, midPoint, videoLength],
        connect: [false, true, true, false],
        range: {
            'min': 0,
            'max': videoLength
        }
    });
    videoLength = Math.floor(videoLength / fps);
    midPoint = Math.floor(midPoint / fps);

    endTimeLabel.innerHTML = formatSeconds(videoLength);
    nowTimeLabel.innerHTML = formatSeconds(midPoint);
    connectSlider2.noUiSlider.on('set', updateSliderElements);
    connectSlider2.noUiSlider.on('slide', updateSliderTimes);
    sliderLoaded = true;
}

function formatSeconds(seconds) {
    let minutes = Math.floor(seconds / 60);
    let remainingSeconds = seconds % 60;
    (remainingSeconds < 10) ? remainingSeconds = '0' + remainingSeconds: remainingSeconds = remainingSeconds.toString();
    if (minutes < 60) {
        let hours = Math.floor(minutes / 60);
        let remainingMinutes = minutes % 60;
        (remainingMinutes < 10) ? remainingMinutes = '0' + remainingMinutes: remainingMinutes = remainingMinutes.toString();
        return hours + ':' + remainingMinutes + ':' + remainingSeconds;
    }
    return minutes + ':' + remainingSeconds;
}

function updateSliderElements(values, handle, unencoded, tap, positions, noUiSlider) {
    console.log('updateSliderElements', values, handle);
    // Convert strings from values to floats
    let startTime = Math.floor(values[0]);
    let nowTime = Math.floor(values[1]);
    let endTime = Math.floor(values[2]);
    let start = document.getElementById('startTimeLabel');
    let end = document.getElementById('endTimeLabel');
    let now = document.getElementById('nowTimeLabel');

    let startNumber = getRealElement('start_time', true);
    let endNumber = getRealElement('end_time', true);
    let nowNumber = getRealElement('current_time', true);
    let fpsNumber = getRealElement('video_fps', true);
    let lastStartTime = startNumber.value;
    let lastNowTime = nowNumber.value;
    let lastEndTime = endNumber.value;
    startNumber.value = startTime;
    nowNumber.value = nowTime;
    endNumber.value = endTime;
    let times = [lastStartTime, lastNowTime, lastEndTime];
    let idx = 0;
    [startNumber, nowNumber, endNumber].forEach(el => {
        if (el.value !== times[idx]) {
            el.dispatchEvent(new Event('input', { 'bubbles': true }));
        }
        idx++;
    });

    let fps = fpsNumber.value;
    startTime = Math.floor(startTime / fps);
    endTime = Math.floor(endTime / fps);
    nowTime = Math.floor(nowTime / fps);
    start.innerHTML = formatSeconds(startTime);
    end.innerHTML = formatSeconds(endTime);
    now.innerHTML = formatSeconds(nowTime);
}

function updateSliderTimes(values, handle, unencoded, tap, positions, noUiSlider) {
    console.log('updateSliderTimes', values, handle);
    // Convert strings from values to floats
    let startTime = Math.floor(values[0]);
    let nowTime = Math.floor(values[1]);
    let endTime = Math.floor(values[2]);
    let start = document.getElementById('startTimeLabel');
    let end = document.getElementById('endTimeLabel');
    let now = document.getElementById('nowTimeLabel');

    let fpsNumber = getRealElement('video_fps', true);
    let fps = fpsNumber.value;
    startTime = Math.floor(startTime / fps);
    endTime = Math.floor(endTime / fps);
    nowTime = Math.floor(nowTime / fps);
    start.innerHTML = formatSeconds(startTime);
    end.innerHTML = formatSeconds(endTime);
    now.innerHTML = formatSeconds(nowTime);
}

function toggleFullscreen() {
    console.log('toggleFullscreen', isActive);
    if (isActive) {
        // If active is true, enumerate openLables and add the .open class to each element
        openLabels.forEach((label) => {
            label.classList.add('open');
            // Get the sibling div and remove display: none!important
            let sibling = label.nextElementSibling;
            sibling.style.display = 'block';
        });
        openLabels = [];
        isActive = false;
    } else {
        openLabels = document.querySelectorAll('.label-wrap.open');
        openLabels.forEach((label) => {
            label.classList.remove('open');
            // Get the sibling div and add display: none!important
            let sibling = label.nextElementSibling;
            sibling.style.display = 'none';

        });
        isActive = true;
    }
    let output = filterArgs(0, arguments);
    console.log('toggleFullscreen', isActive, output);
    return output;
}

function downloadImage() {
    //.thumbnail-item.selected
    let selectedThumbnail = document.querySelector('.thumbnail-item.selected');
    let args;
    if (selectedThumbnail) {
        let img = selectedThumbnail.querySelector('img');
        if (img) {
            let url = img.src;
            let filename = url.split('/').pop();
            let path = url.split('/').slice(0, -1).join('/');
            args = {
                url: url,
                filename: filename,
                path: path
            };
            console.log('downloadImage', args);
        }
    } else {
        let gallery1 = document.getElementById('gallery1');
        // Get the second img, which is a child somewhere in gallery1
        let img = gallery1.querySelectorAll('img');
        if (img.length > 1) {
            let url = img[1].src;
            let filename = url.split('/').pop();
            let path = url.split('/').slice(0, -1).join('/');
            args = {
                url: url,
                filename: filename,
                path: path
            };
            console.log('downloadImage', args);
        }

    }
    if (args.hasOwnProperty('url')) {
            console.log('downloadImage (url)', args.url, args.filename);
            let url = args.url;
            if (url.length > 0) {
                fetch(url)
                    .then(response => response.blob())
                    .then(blob => {
                        let link = document.createElement('a');
                        link.href = window.URL.createObjectURL(blob);
                        if (args.hasOwnProperty('path')) {
                            let path = args.path;
                            let filename = path.split('/').pop();
                            if (filename.length > 0) {
                                link.download = filename;
                            }
                        }
                        // This is necessary as link.click() does not work on the latest firefox
                        link.style.display = 'none';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        window.URL.revokeObjectURL(link.href); // Clean up the URL object
                    })
                    .catch(console.error);
            }
        }

}


document.addEventListener('keydown', (event) => {
    if (isActive && event.key === 'Escape') {
        console.log('Escape key pressed');
        fullscreenButton = getRealElement('fullscreen_button');
        // Click the fullscreen button to close the fullscreen mode
        fullscreenButton.click();
    }
});

document.addEventListener('click', (event) => {
    // If the target has the class "image-button"
    if (event.target.classList.contains('image-button')) {
        // Get the child img object
        let img = event.target.querySelector('img');
        // Toggle the "fullscreen" class on the img object
        img.classList.toggle('full_screen');
    }
});

// Document ready
document.addEventListener('DOMContentLoaded', () => {
    // Get the fullscreen button
    console.log('DOMContentLoaded');
});