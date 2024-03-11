let isActive = false;
let openLabels = [];
let fullscreenButton;


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