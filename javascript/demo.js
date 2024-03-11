let isActive = false;
let openLabels = [];
let fullscreenButton;


function getRealElement(selector) {
    let elem = gradioApp().getElementById(selector);
    if (elem) {
        let child = elem.querySelector('#' + selector);
        if (child) {
            return child;
        } else {
            return elem;
        }
    }
    return elem;
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
    console.log('downloadImage', arguments);
}

document.addEventListener('keydown', (event) => {
    if (isActive && event.key === 'Escape') {
        fullscreenButton = getRealElement('fullscreen_button');
        // Click the fullscreen button to close the fullscreen mode
        fullscreenButton.click();
    }
});

// Document ready
document.addEventListener('DOMContentLoaded', () => {
    // Get the fullscreen button
    console.log('DOMContentLoaded');
});